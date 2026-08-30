// macocr — minimal Vision-framework OCR CLI used by epicat.
//
// Usage: macocr <image.png> [--langs zh-Hans,en-US] [--fast] [--min-conf 0.0]
// Prints JSON: {"width":W,"height":H,"lines":[{"text":..,"conf":..,"x":..,"y":..,"w":..,"h":..}]}
// Box coordinates are pixels with a top-left origin.

import Foundation
import Vision
import CoreGraphics
import ImageIO

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(2)
}

var path: String? = nil
var langs = ["zh-Hans", "en-US"]
var accurate = true
var minConf: Float = 0.0

var args = Array(CommandLine.arguments.dropFirst())
var i = 0
while i < args.count {
    switch args[i] {
    case "--langs":
        i += 1
        guard i < args.count else { fail("--langs needs a value") }
        langs = args[i].split(separator: ",").map(String.init)
    case "--fast":
        accurate = false
    case "--min-conf":
        i += 1
        guard i < args.count, let v = Float(args[i]) else { fail("--min-conf needs a number") }
        minConf = v
    default:
        path = args[i]
    }
    i += 1
}

guard let path = path else { fail("usage: macocr <image> [--langs zh-Hans,en-US] [--fast]") }

let url = URL(fileURLWithPath: path)
guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
      let cgImage = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
    fail("cannot read image: \(path)")
}

let W = CGFloat(cgImage.width)
let H = CGFloat(cgImage.height)

let request = VNRecognizeTextRequest()
request.recognitionLevel = accurate ? .accurate : .fast
request.recognitionLanguages = langs
request.usesLanguageCorrection = false

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
do {
    try handler.perform([request])
} catch {
    fail("OCR failed: \(error)")
}

struct Line: Codable {
    let text: String
    let conf: Float
    let x: Int, y: Int, w: Int, h: Int
}
struct Result: Codable {
    let width: Int, height: Int
    let lines: [Line]
}

var lines: [Line] = []
for obs in (request.results ?? []) {
    guard let cand = obs.topCandidates(1).first else { continue }
    if cand.confidence < minConf { continue }
    // Vision boxes are normalized, bottom-left origin.
    let bb = obs.boundingBox
    let x = bb.origin.x * W
    let w = bb.size.width * W
    let h = bb.size.height * H
    let y = (1.0 - bb.origin.y - bb.size.height) * H
    lines.append(Line(text: cand.string,
                      conf: cand.confidence,
                      x: Int(x.rounded(.down)), y: Int(y.rounded(.down)),
                      w: Int(w.rounded(.up)), h: Int(h.rounded(.up))))
}

let out = Result(width: cgImage.width, height: cgImage.height, lines: lines)
let enc = JSONEncoder()
enc.outputFormatting = [.sortedKeys]
FileHandle.standardOutput.write(try! enc.encode(out))
FileHandle.standardOutput.write("\n".data(using: .utf8)!)
