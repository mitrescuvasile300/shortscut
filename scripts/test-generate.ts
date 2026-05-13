/**
 * Test just the "Generate Shorts" button on an existing job
 * Navigates to the most recent completed job and clicks Generate
 */
import { runTest, type PageHelper } from "./auth";

const TEST_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ";

runTest("Generate Shorts Test", async (helper: PageHelper) => {
  const page = helper.page;

  // First, check if there's already a completed job
  console.log("\n📋 Step 1: Check for existing completed job...");
  await helper.goto("/");
  await page.waitForTimeout(2000);

  let jobUrl = "";

  // Look for a job link on the dashboard
  const jobLinks = page.locator("a[href*='/job/']");
  const count = await jobLinks.count();
  console.log(`  Found ${count} job links on dashboard`);

  if (count > 0) {
    // Click the first job
    const href = await jobLinks.first().getAttribute("href");
    console.log(`  Navigating to: ${href}`);
    await jobLinks.first().click();
    await page.waitForTimeout(2000);
    jobUrl = page.url();
  }

  // Check if we're on a completed job with the generate button
  let content = await helper.getPageContent();
  const hasGenerateBtn = content.includes("Generează") && content.includes("Shorts");

  if (!hasGenerateBtn) {
    console.log("  No completed job found. Creating new one...");
    await helper.goto("/new");
    await page.waitForTimeout(1000);

    const urlInput = page.locator("input#videoUrl");
    await urlInput.fill(TEST_VIDEO_URL);
    await page.waitForTimeout(500);

    const submitBtn = page.locator("button[type='submit']");
    await submitBtn.click();
    await page.waitForTimeout(2000);

    // Wait for completion
    const maxWait = 180000;
    let elapsed = 0;
    while (elapsed < maxWait) {
      await page.waitForTimeout(5000);
      elapsed += 5000;
      content = await helper.getPageContent();
      if (content.includes("Generează Shorts")) {
        console.log(`  ✅ Job completed (${elapsed / 1000}s)`);
        break;
      }
      if (content.includes("Eroare") || content.includes("failed")) {
        throw new Error("Job failed!");
      }
      const status = content.match(/(downloading|transcribing|analyzing)/i);
      console.log(`  Status: ${status?.[1] || "processing"} (${elapsed / 1000}s)`);
    }
  }

  // ===== STEP 2: Click Generate Shorts =====
  console.log("\n🎬 Step 2: Clicking Generate Shorts...");
  await helper.screenshot("before-generate.png");

  // Listen for ALL console messages to track progress
  page.on("console", (msg) => {
    const text = msg.text();
    if (text.includes("[ffmpeg]") || text.includes("[faceDetection]") || 
        text.includes("Download") || text.includes("Error") || 
        text.includes("error") || text.includes("CORS") ||
        text.includes("fetch") || text.includes("proxy")) {
      console.log(`  [browser ${msg.type()}] ${text.substring(0, 300)}`);
    }
  });

  // Track page errors
  page.on("pageerror", (err) => {
    console.log(`  [PAGE ERROR] ${err.message.substring(0, 300)}`);
  });

  // Track failed requests
  page.on("requestfailed", (req) => {
    console.log(`  [REQUEST FAILED] ${req.url().substring(0, 150)} - ${req.failure()?.errorText}`);
  });

  const generateBtn = page.locator("button").filter({ hasText: /Generează.*Shorts/ }).first();
  if (!(await generateBtn.isVisible())) {
    throw new Error("Generate button not visible!");
  }

  await generateBtn.click();
  console.log("  Button clicked, waiting for processing...");

  // Wait for actual results - look for "Descarcă" (download) buttons specifically 
  const maxGenWait = 600000; // 10 min
  let genElapsed = 0;
  let lastProgress = "";

  while (genElapsed < maxGenWait) {
    await page.waitForTimeout(3000);
    genElapsed += 3000;

    content = await helper.getPageContent();

    // Check for download buttons (definitive success)
    const downloadBtnCount = await page.locator("button").filter({ hasText: "Descarcă" }).count();
    const videosCount = await page.locator("video").count();

    if (downloadBtnCount > 0 || videosCount > 0) {
      console.log(`  ✅ Generation complete! (${genElapsed / 1000}s) - ${downloadBtnCount} download btns, ${videosCount} videos`);
      break;
    }

    // Check for errors in UI
    if (content.includes("Eroare la Short")) {
      console.log(`  ❌ Error during generation!`);
      await helper.screenshot("gen-error.png");
      // Don't throw - let's capture the error
      break;
    }

    // Show progress from page text
    const progressMatch = content.match(/(Se (?:încarcă|descarcă|detectează|procesează|encodează)[^\n]*)/);
    const progress = progressMatch?.[1] || "";
    if (progress && progress !== lastProgress) {
      console.log(`  Progress: ${progress} (${genElapsed / 1000}s)`);
      lastProgress = progress;
    }

    // Also check for "done" message in page
    const doneMatch = content.match(/(\d+\/\d+ shorts generate)/);
    if (doneMatch) {
      console.log(`  ✅ Done: ${doneMatch[1]} (${genElapsed / 1000}s)`);
      break;
    }
  }

  if (genElapsed >= maxGenWait) {
    console.log("  ⚠️ Timeout!");
  }

  await helper.screenshot("after-generate.png");

  console.log("\n📝 Final page state (first 2000 chars):");
  const finalContent = await helper.getPageContent();
  console.log(finalContent.substring(0, 2000));
});
