import { runTest } from "./auth";
import fs from "fs";

const COOKIES_FILE = "/tmp/raw_cookies.txt";
const TEST_VIDEO = "https://www.youtube.com/watch?v=vsp69jYlYsg";

runTest("Cookie test - save cookies and process bot-protected video", async (page, helper) => {
  const cookies = fs.readFileSync(COOKIES_FILE, "utf-8");
  console.log(`[Test] Loaded ${cookies.split("\n").filter(l => l.trim() && !l.startsWith("#")).length} cookie lines`);
  
  // Step 1: Go to Settings and save cookies
  console.log("[Test] Step 1: Navigating to Settings...");
  await page.goto(`${process.env.APP_URL}/settings`);
  await page.waitForLoadState("networkidle");
  await helper.screenshot("settings-page");
  
  // Find and fill the YouTube Cookies textarea
  // First, click to show the cookies input area
  const cookieSection = page.locator("text=YouTube Cookies").first();
  await cookieSection.waitFor({ timeout: 10000 });
  console.log("[Test] Found YouTube Cookies section");
  
  // Click the show/expand area to reveal textarea
  const showArea = page.locator("text=Nu sunt setate").first().or(page.locator("text=●●●●●●●●").first());
  try {
    await showArea.click({ timeout: 5000 });
    console.log("[Test] Clicked to show cookies input");
  } catch {
    // Maybe already visible, try to find eye icon
    const eyeBtn = cookieSection.locator("..").locator("button").first();
    await eyeBtn.click({ timeout: 5000 });
    console.log("[Test] Clicked eye button to show cookies");
  }
  
  await page.waitForTimeout(500);
  
  // Fill the textarea
  const textarea = page.locator("textarea#youtubeCookies");
  await textarea.waitFor({ timeout: 5000 });
  await textarea.fill(cookies);
  console.log("[Test] Filled cookies textarea");
  await helper.screenshot("settings-cookies-filled");
  
  // Save settings
  const saveBtn = page.locator("button:has-text('Salvează')");
  await saveBtn.click();
  await page.waitForTimeout(2000);
  console.log("[Test] Saved settings");
  await helper.screenshot("settings-saved");
  
  // Step 2: Navigate to dashboard and create a new job with the problematic video
  console.log("[Test] Step 2: Creating job with bot-protected video...");
  await page.goto(`${process.env.APP_URL}/`);
  await page.waitForLoadState("networkidle");
  await page.waitForTimeout(1000);
  
  // Find the URL input and submit a job
  const urlInput = page.locator('input[placeholder*="youtube"]').first()
    .or(page.locator('input[type="url"]').first())
    .or(page.locator('input[name="videoUrl"]').first());
  
  try {
    await urlInput.waitFor({ timeout: 5000 });
  } catch {
    // Maybe it's a different selector
    console.log("[Test] Looking for URL input...");
    const allInputs = await page.locator("input").all();
    for (const input of allInputs) {
      const placeholder = await input.getAttribute("placeholder");
      const type = await input.getAttribute("type");
      console.log(`  Input: placeholder="${placeholder}" type="${type}"`);
    }
    throw new Error("Could not find URL input");
  }
  
  await urlInput.fill(TEST_VIDEO);
  console.log("[Test] Filled video URL");
  
  // Submit
  const submitBtn = page.locator("button[type='submit']").first()
    .or(page.locator("button:has-text('Adaugă')").first())
    .or(page.locator("button:has-text('Creează')").first());
  await submitBtn.click();
  console.log("[Test] Submitted job");
  
  await page.waitForTimeout(3000);
  await helper.screenshot("job-submitted");
  
  // Wait for job to go to processing - look for a job detail page or status changes
  // The page should redirect to the job detail
  console.log("[Test] Step 3: Waiting for job to process...");
  
  // Wait up to 5 minutes for the job to go through processing
  const maxWait = 300000; // 5 minutes
  const startTime = Date.now();
  let lastStatus = "";
  let succeeded = false;
  
  while (Date.now() - startTime < maxWait) {
    await page.waitForTimeout(5000);
    
    const content = await helper.content();
    
    // Check for various status indicators
    if (content.includes("Se generează shorts") || content.includes("generating")) {
      if (lastStatus !== "generating") {
        console.log(`[Test] Status: generating (${Math.round((Date.now() - startTime) / 1000)}s)`);
        lastStatus = "generating";
        await helper.screenshot("job-generating");
      }
    } else if (content.includes("Se analizează") || content.includes("analyzing")) {
      if (lastStatus !== "analyzing") {
        console.log(`[Test] Status: analyzing (${Math.round((Date.now() - startTime) / 1000)}s)`);
        lastStatus = "analyzing";
      }
    } else if (content.includes("Se transcrie") || content.includes("transcribing")) {
      if (lastStatus !== "transcribing") {
        console.log(`[Test] Status: transcribing (${Math.round((Date.now() - startTime) / 1000)}s)`);
        lastStatus = "transcribing";
      }
    } else if (content.includes("Se descarcă") || content.includes("downloading")) {
      if (lastStatus !== "downloading") {
        console.log(`[Test] Status: downloading (${Math.round((Date.now() - startTime) / 1000)}s)`);
        lastStatus = "downloading";
      }
    } else if (content.includes("Finalizat") || content.includes("completed") || content.includes("Descarcă")) {
      console.log(`[Test] ✅ Job completed! (${Math.round((Date.now() - startTime) / 1000)}s)`);
      succeeded = true;
      await helper.screenshot("job-completed");
      break;
    } else if (content.includes("Eroare") || content.includes("error") || content.includes("eșuat") || content.includes("failed")) {
      console.log(`[Test] ❌ Job failed! (${Math.round((Date.now() - startTime) / 1000)}s)`);
      await helper.screenshot("job-failed");
      // Print the error
      const errorMatch = content.match(/(?:Eroare|Error|eșuat|failed)[^.]*\./i);
      if (errorMatch) console.log(`[Test] Error: ${errorMatch[0]}`);
      // Print the whole page content for debugging
      console.log(`[Test] Page content (first 2000 chars): ${content.substring(0, 2000)}`);
      break;
    }
    
    // Also check if we're still on dashboard (job might have error before redirect)
    if (content.includes("Nu am putut accesa")) {
      console.log(`[Test] ❌ YouTube access error detected!`);
      console.log(`[Test] Content snippet: ${content.substring(content.indexOf("Nu am putut"), content.indexOf("Nu am putut") + 200)}`);
      await helper.screenshot("youtube-access-error");
      break;
    }
  }
  
  if (!succeeded && Date.now() - startTime >= maxWait) {
    console.log("[Test] ⏰ Timeout waiting for job completion");
    await helper.screenshot("job-timeout");
  }
  
  // Final screenshot
  await helper.screenshot("final-state");
  console.log("[Test] Done!");
});
