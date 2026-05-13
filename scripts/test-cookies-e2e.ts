import { runTest } from "./auth";
import fs from "fs";

const COOKIES_FILE = "/tmp/raw_cookies.txt";
const TEST_VIDEO = "https://www.youtube.com/watch?v=vsp69jYlYsg";

function sleep(ms: number): Promise<void> {
  return new Promise(r => setTimeout(r, ms));
}

runTest("Cookie E2E — save cookies + process bot-protected video", async (helper) => {
  const page = helper.page;
  const cookies = fs.readFileSync(COOKIES_FILE, "utf-8");
  console.log(`[Test] Loaded ${cookies.split("\n").filter(l => l.trim() && !l.startsWith("#")).length} cookie lines`);

  // Step 1: Save cookies via Settings page
  console.log("[Test] Step 1: Save cookies in Settings...");
  await helper.goto("/settings");
  await sleep(2000);

  // Click the masked area to show textarea  
  try {
    const maskedArea = page.locator("text=Nu sunt setate").or(page.locator("text=●●●●●●●●")).first();
    await maskedArea.click({ timeout: 3000 });
    console.log("[Test] Clicked masked area");
  } catch {
    // Try the eye button
    try {
      const eyeButtons = await page.locator("button").all();
      for (const btn of eyeButtons) {
        const html = await btn.innerHTML().catch(() => "");
        if (html.includes("eye") || html.includes("Eye")) {
          await btn.click();
          console.log("[Test] Clicked eye button");
          break;
        }
      }
    } catch {}
  }

  await sleep(500);

  // Fill textarea
  let textarea = page.locator("textarea").first();
  try {
    await textarea.waitFor({ timeout: 3000 });
    await textarea.fill(cookies);
    console.log("[Test] Filled cookies textarea");
  } catch {
    console.log("[Test] WARNING: Could not find textarea");
  }

  // Save
  const saveBtn = page.locator("button:has-text('Salvează')");
  await saveBtn.click();
  await sleep(2000);
  console.log("[Test] Settings saved");
  await helper.screenshot("settings-saved.png");

  // Step 2: Create job with bot-protected video
  console.log("[Test] Step 2: Creating job...");
  await helper.goto("/new");
  await sleep(2000);

  // Look for URL input
  const inputs = await page.locator("input").all();
  let urlInput = null;
  for (const input of inputs) {
    const ph = await input.getAttribute("placeholder") || "";
    const type = await input.getAttribute("type") || "";
    if (ph.toLowerCase().includes("youtube") || ph.toLowerCase().includes("url") || type === "url") {
      urlInput = input;
      break;
    }
  }
  if (!urlInput) urlInput = page.locator("input").first();

  await urlInput.fill(TEST_VIDEO);
  console.log("[Test] Filled URL: " + TEST_VIDEO);
  await sleep(500);

  // Submit
  const submitBtns = await page.locator("button[type='submit'], button:has-text('Procesează'), button:has-text('Adaugă'), button:has-text('Creează')").all();
  for (const btn of submitBtns) {
    const disabled = await btn.isDisabled().catch(() => true);
    if (!disabled) {
      const text = await btn.textContent();
      console.log(`[Test] Clicking submit: "${text}"`);
      await btn.click();
      break;
    }
  }

  await sleep(3000);
  await helper.screenshot("job-submitted.png");

  // Step 3: Wait for processing
  console.log("[Test] Step 3: Waiting for processing (max 5min)...");
  const maxWait = 300000;
  const start = Date.now();
  let lastStatus = "";
  let succeeded = false;

  while (Date.now() - start < maxWait) {
    await sleep(5000);
    const content = await helper.getPageContent();

    const statusChecks = [
      { match: ["Se descarcă", "downloading"], name: "downloading" },
      { match: ["Se transcrie", "transcribing"], name: "transcribing" },
      { match: ["Se analizează", "analyzing"], name: "analyzing" },
      { match: ["Se generează", "generating"], name: "generating" },
      { match: ["Finalizat", "completed", "Descarcă Toate"], name: "completed" },
      { match: ["Eroare", "error", "eșuat", "failed", "Nu am putut"], name: "error" },
    ];

    for (const check of statusChecks) {
      if (check.match.some(m => content.includes(m))) {
        if (lastStatus !== check.name) {
          const elapsed = Math.round((Date.now() - start) / 1000);
          console.log(`[Test] Status: ${check.name} (${elapsed}s)`);
          lastStatus = check.name;
          await helper.screenshot(`status-${check.name}.png`);

          if (check.name === "completed") succeeded = true;
          if (check.name === "error") {
            for (const m of check.match) {
              const idx = content.indexOf(m);
              if (idx >= 0) {
                console.log(`[Test] Error: ${content.substring(Math.max(0, idx - 50), idx + 300)}`);
                break;
              }
            }
          }
        }
        break;
      }
    }

    if (succeeded || lastStatus === "error") break;
  }

  if (!succeeded && lastStatus !== "error") {
    console.log(`[Test] ⏰ Timeout after ${Math.round((Date.now() - start) / 1000)}s (last: ${lastStatus})`);
    await helper.screenshot("timeout.png");
  }

  await helper.screenshot("final.png");
  console.log(`[Test] Result: ${succeeded ? "✅ PASSED" : "❌ FAILED"}`);
  if (!succeeded) process.exit(1);
});
