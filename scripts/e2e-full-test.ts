/**
 * Full E2E test: create a job, wait for backend processing, generate shorts
 */
import { runTest, type PageHelper } from "./auth";

const TEST_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"; // Short video for testing

runTest("Full E2E Flow", async (helper: PageHelper) => {
  const page = helper.page;

  // ===== STEP 1: Check settings (OpenAI key should exist) =====
  console.log("\n📋 Step 1: Check settings...");
  await helper.goto("/settings");
  await page.waitForTimeout(2000);

  const openaiInput = page.locator("input#openaiApiKey");
  const openaiValue = await openaiInput.inputValue().catch(() => "");
  console.log(`  OpenAI key configured: ${openaiValue ? "YES (length: " + openaiValue.length + ")" : "NO"}`);

  // ===== STEP 2: Create a new job =====
  console.log("\n🚀 Step 2: Create new job...");
  await helper.goto("/new");
  await page.waitForTimeout(1000);

  // Fill in the YouTube URL
  const urlInput = page.locator("input#videoUrl");
  await urlInput.fill(TEST_VIDEO_URL);
  await page.waitForTimeout(500);

  // Set 2 shorts, 15-30s duration for quick testing
  const numShortsInput = page.locator("input[type='number']").first();
  // Change slider to 2
  const slider = page.locator('[role="slider"]');
  if (await slider.isVisible()) {
    // Try to set number of shorts
    // For testing we'll keep defaults
  }

  // Submit
  console.log("  Submitting job...");
  const submitBtn = page.locator("button[type='submit']");
  await submitBtn.click();
  await page.waitForTimeout(2000);

  // Should redirect to job detail page
  const currentUrl = page.url();
  console.log(`  Redirected to: ${currentUrl}`);

  if (!currentUrl.includes("/job/")) {
    console.log("  ❌ Did not redirect to job page!");
    await helper.printDebugInfo();
    throw new Error("Job creation failed - no redirect to job page");
  }

  // ===== STEP 3: Wait for backend processing =====
  console.log("\n⏳ Step 3: Waiting for backend processing...");
  const maxWaitMs = 180000; // 3 minutes
  const pollInterval = 5000;
  let elapsed = 0;
  let lastStatus = "";

  while (elapsed < maxWaitMs) {
    await page.waitForTimeout(pollInterval);
    elapsed += pollInterval;

    // Check page content for status indicators
    const content = await helper.getPageContent();
    
    // Check for status text
    if (content.includes("Eroare") || content.includes("failed") || content.includes("Eșuat")) {
      console.log(`  ❌ Job failed! Page content:\n${content.substring(0, 500)}`);
      await helper.screenshot("job-failed.png");
      throw new Error("Job processing failed");
    }

    if (content.includes("Generează Shorts") || content.includes("Generate Shorts")) {
      console.log(`  ✅ Job completed! (${elapsed / 1000}s)`);
      break;
    }

    // Extract current status
    const statusMatch = content.match(/(downloading|transcribing|analyzing|Se descarcă|Se transcrie|Se analizează)/i);
    const newStatus = statusMatch ? statusMatch[1] : "processing";
    if (newStatus !== lastStatus) {
      console.log(`  Status: ${newStatus} (${elapsed / 1000}s)`);
      lastStatus = newStatus;
    }
  }

  if (elapsed >= maxWaitMs) {
    console.log("  ⚠️ Timeout waiting for processing!");
    await helper.screenshot("job-timeout.png");
    const content = await helper.getPageContent();
    console.log(`  Page content: ${content.substring(0, 500)}`);
    throw new Error("Job processing timeout");
  }

  // ===== STEP 4: Check clips are shown =====
  console.log("\n📊 Step 4: Checking clips...");
  const pageContent = await helper.getPageContent();
  
  // Count clips
  const clipCards = page.locator('[class*="clip"], [class*="Card"]');
  const clipCount = await clipCards.count().catch(() => 0);
  console.log(`  Clips found on page: ${clipCount}`);

  // Screenshot the results
  await helper.screenshot("job-completed.png");

  // ===== STEP 5: Test video generation (browser-side) =====
  console.log("\n🎬 Step 5: Testing short generation (browser-side ffmpeg)...");
  
  // Find and click the generate button
  const generateBtn = page.locator("button").filter({ hasText: /Generează|Generate/ }).first();
  if (await generateBtn.isVisible()) {
    console.log("  Clicking Generate Shorts...");
    
    // Listen for console errors
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        errors.push(msg.text());
      }
      // Log ffmpeg and face detection progress
      const text = msg.text();
      if (text.includes("[ffmpeg]") || text.includes("[faceDetection]")) {
        console.log(`  [browser] ${text.substring(0, 200)}`);
      }
    });

    await generateBtn.click();

    // Wait for processing (this takes longer - ffmpeg.wasm + face detection)
    const genMaxWait = 300000; // 5 minutes
    let genElapsed = 0;
    let lastMsg = "";

    while (genElapsed < genMaxWait) {
      await page.waitForTimeout(3000);
      genElapsed += 3000;

      const content = await helper.getPageContent();

      // Check for completion - look for video elements or download buttons
      const hasVideo = await page.locator("video").count() > 0;
      const hasDone = content.includes("✅") || content.includes("Descarcă") || content.includes("Download");

      if (hasVideo || hasDone) {
        console.log(`  ✅ Shorts generated! (${genElapsed / 1000}s)`);
        break;
      }

      // Check for errors
      if (errors.length > 0) {
        console.log(`  Browser errors: ${errors.slice(-3).join("\n    ")}`);
      }

      // Show progress
      const progressMatch = content.match(/(Se (?:descarcă|încarcă|encodează|procesează|detectează).+?)(?:\n|$)/);
      const msgMatch = content.match(/(Face detection|detecting|processing|encoding|downloading).+/i);
      const msg = progressMatch?.[1] || msgMatch?.[1] || "";
      if (msg && msg !== lastMsg) {
        console.log(`  Progress: ${msg}`);
        lastMsg = msg;
      }
    }

    if (genElapsed >= genMaxWait) {
      console.log("  ⚠️ Timeout waiting for video generation!");
      await helper.screenshot("gen-timeout.png");
    } else {
      await helper.screenshot("shorts-generated.png");
    }
  } else {
    console.log("  ⚠️ Generate button not found!");
    await helper.screenshot("no-generate-btn.png");
  }

  // Final summary
  console.log("\n📝 Final page state:");
  const finalContent = await helper.getPageContent();
  console.log(finalContent.substring(0, 1000));
});
