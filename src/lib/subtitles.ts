/**
 * ASS (Advanced SubStation Alpha) subtitle generation
 * Generates styled subtitles for ffmpeg libass burning
 */

export interface SubtitleSegment {
  start: number; // seconds
  end: number;   // seconds
  text: string;
}

function formatAssTime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  const sInt = Math.floor(s);
  const cs = Math.floor((s - sInt) * 100);
  return `${h}:${m.toString().padStart(2, "0")}:${sInt.toString().padStart(2, "0")}.${cs.toString().padStart(2, "0")}`;
}

/**
 * Get transcript segments that overlap with a clip's time range
 */
export function getSegmentsForClip(
  allSegments: SubtitleSegment[],
  clipStart: number,
  clipEnd: number
): SubtitleSegment[] {
  return allSegments
    .filter(seg => seg.end > clipStart && seg.start < clipEnd)
    .map(seg => ({
      start: Math.max(0, seg.start - clipStart),
      end: Math.min(clipEnd - clipStart, seg.end - clipStart),
      text: seg.text.replace(/\n/g, " ").trim(),
    }))
    .filter(seg => seg.text.length > 0);
}

/**
 * Generate an ASS subtitle file content
 * Style: white text with black outline, positioned at bottom center
 */
export function generateAssSubtitles(
  segments: SubtitleSegment[],
  width = 1080,
  height = 1920
): string {
  const fontSize = 48;
  const marginV = 120; // vertical margin from bottom
  const outlineSize = 3;

  let ass = `[Script Info]
Title: ShortsCut Subtitles
ScriptType: v4.00+
PlayResX: ${width}
PlayResY: ${height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,${fontSize},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,${outlineSize},1,2,20,20,${marginV},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
`;

  for (const seg of segments) {
    // Escape special ASS characters and wrap long lines
    let text = seg.text
      .replace(/\\/g, "\\\\")
      .replace(/{/g, "\\{")
      .replace(/}/g, "\\}");

    // Word wrap: split into lines of ~30 chars max for vertical video
    const words = text.split(" ");
    const lines: string[] = [];
    let currentLine = "";
    for (const word of words) {
      if (currentLine.length + word.length + 1 > 30 && currentLine.length > 0) {
        lines.push(currentLine);
        currentLine = word;
      } else {
        currentLine = currentLine ? `${currentLine} ${word}` : word;
      }
    }
    if (currentLine) lines.push(currentLine);
    text = lines.join("\\N"); // ASS line break

    ass += `Dialogue: 0,${formatAssTime(seg.start)},${formatAssTime(seg.end)},Default,,0,0,0,,${text}\n`;
  }

  return ass;
}
