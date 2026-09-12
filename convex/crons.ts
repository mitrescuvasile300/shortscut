import { cronJobs } from "convex/server";
import { internal } from "./_generated/api";

const crons = cronJobs();

// Re-attach jobs whose VPS polling/pull action died (see vpsPipeline.watchdog).
crons.interval("vps job watchdog", { minutes: 5 }, internal.vpsPipeline.watchdog, {});

export default crons;
