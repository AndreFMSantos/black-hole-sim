import AppKit
import Foundation

guard CommandLine.arguments.count >= 5 else {
  fputs("Usage: render_overlay.swift <out.png> <tag> <title> <subtitle>\n", stderr)
  exit(1)
}

let outPath = CommandLine.arguments[1]
let tag = CommandLine.arguments[2]
let title = CommandLine.arguments[3]
let subtitle = CommandLine.arguments[4]

let width = 1280
let height = 720

let rep = NSBitmapImageRep(
  bitmapDataPlanes: nil,
  pixelsWide: width,
  pixelsHigh: height,
  bitsPerSample: 8,
  samplesPerPixel: 4,
  hasAlpha: true,
  isPlanar: false,
  colorSpaceName: .deviceRGB,
  bytesPerRow: 0,
  bitsPerPixel: 0
)!

NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)

NSColor.clear.set()
NSRect(x: 0, y: 0, width: width, height: height).fill()

// Lower-third card (AppKit origin is bottom-left)
let cardRect = NSRect(x: 36, y: 32, width: 690, height: 74)
let cardPath = NSBezierPath(roundedRect: cardRect, xRadius: 12, yRadius: 12)
NSColor(calibratedRed: 0.04, green: 0.05, blue: 0.08, alpha: 0.78).setFill()
cardPath.fill()
NSColor(calibratedRed: 0.95, green: 0.65, blue: 0.25, alpha: 0.45).setStroke()
cardPath.lineWidth = 1.5
cardPath.stroke()

// Version pill
let pillRect = NSRect(x: 50, y: 58, width: 88, height: 30)
let pillPath = NSBezierPath(roundedRect: pillRect, xRadius: 7, yRadius: 7)
NSColor(calibratedRed: 0.95, green: 0.58, blue: 0.16, alpha: 0.95).setFill()
pillPath.fill()

let tagStyle = NSMutableParagraphStyle()
tagStyle.alignment = .center
let tagAttrs: [NSAttributedString.Key: Any] = [
  .font: NSFont.monospacedSystemFont(ofSize: 15, weight: .bold),
  .foregroundColor: NSColor(calibratedRed: 0.06, green: 0.06, blue: 0.08, alpha: 1.0),
  .paragraphStyle: tagStyle,
]
(tag as NSString).draw(in: NSRect(x: 50, y: 63, width: 88, height: 22), withAttributes: tagAttrs)

// Title
let titleAttrs: [NSAttributedString.Key: Any] = [
  .font: NSFont.systemFont(ofSize: 20, weight: .semibold),
  .foregroundColor: NSColor(calibratedWhite: 0.97, alpha: 1.0),
]
(title as NSString).draw(at: NSPoint(x: 152, y: 61), withAttributes: titleAttrs)

// Subtitle
let subAttrs: [NSAttributedString.Key: Any] = [
  .font: NSFont.monospacedSystemFont(ofSize: 13, weight: .regular),
  .foregroundColor: NSColor(calibratedRed: 0.78, green: 0.82, blue: 0.88, alpha: 0.95),
]
(subtitle as NSString).draw(at: NSPoint(x: 52, y: 40), withAttributes: subAttrs)

NSGraphicsContext.restoreGraphicsState()

guard let png = rep.representation(using: .png, properties: [:]) else {
  fputs("Failed to encode PNG\n", stderr)
  exit(1)
}
try png.write(to: URL(fileURLWithPath: outPath))
