import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { chromium } from "playwright";

const START_URL = "https://bus.go.kr/";
const QUERY = "A148";
const outputDir = path.resolve(
  process.env.PROBE_OUTPUT || "a148_public_page_probe_output",
);

const allowedBodyHosts = new Set([
  "bus.go.kr",
  "www.bus.go.kr",
  "m.bus.go.kr",
  "topis.seoul.go.kr",
]);
const interestingUrl = /bus|route|line|position|location|arrival|station|stop|veh/i;
const structuredContent = /json|xml|javascript|text\/plain/i;
const sensitiveParameter = /key|token|secret|auth|session|cookie|csrf|credential/i;
const locationField = /\b(?:tmX|tmY|posX|posY|gpsX|gpsY|lat|latitude|lon|lng|longitude|vehId|plainNo|sectOrd)\b/i;
const blockedText = /captcha|access denied|forbidden|unusual traffic|robot|\ub85c\ubd07|\uc790\ub3d9\uc785\ub825|\uc811\uadfc\uc774 \ucc28\ub2e8|\ub85c\uadf8\uc778\uc774 \ud544\uc694/i;

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(path.join(outputDir, "public_responses"), { recursive: true });

function sanitizeUrl(rawUrl) {
  try {
    const url = new URL(rawUrl);
    for (const [name] of url.searchParams) {
      if (sensitiveParameter.test(name)) {
        url.searchParams.set(name, "[redacted]");
      }
    }
    return url.toString();
  } catch {
    return rawUrl.replace(/([?&][^=]*(?:key|token|secret|auth|session|cookie|csrf)[^=]*=)[^&]*/gi, "$1[redacted]");
  }
}

function safeFileName(index, rawUrl, contentType) {
  const url = new URL(rawUrl);
  const stem = `${url.hostname}${url.pathname}`
    .replace(/[^a-zA-Z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 100) || "response";
  const extension = /json/i.test(contentType)
    ? "json"
    : /xml/i.test(contentType)
      ? "xml"
      : "txt";
  return `${String(index).padStart(3, "0")}_${stem}.${extension}`;
}

async function visiblePageText(page) {
  return (await page.locator("body").innerText({ timeout: 10_000 })).slice(0, 1_000_000);
}

async function savePageState(page, label) {
  const text = await visiblePageText(page).catch((error) => `TEXT_CAPTURE_ERROR: ${error.message}`);
  await fs.writeFile(path.join(outputDir, `${label}.txt`), text, "utf8");
  await page.screenshot({
    path: path.join(outputDir, `${label}.png`),
    fullPage: true,
  });
  return text;
}

async function firstVisible(locator) {
  const count = await locator.count();
  for (let index = 0; index < Math.min(count, 30); index += 1) {
    const candidate = locator.nth(index);
    if (await candidate.isVisible().catch(() => false)) {
      return candidate;
    }
  }
  return null;
}

const networkLog = [];
const savedResponses = [];
const responseTasks = [];
let responseIndex = 0;
let browser;
let page;
let initialText = "";
let searchText = "";
let routeText = "";
let searchAttempt = "not_attempted";
let routeClick = "not_attempted";
let fatalError = null;

try {
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    locale: "ko-KR",
    timezoneId: "Asia/Seoul",
    viewport: { width: 1440, height: 1100 },
  });
  page = await context.newPage();

  page.on("response", (response) => {
    const task = (async () => {
      const request = response.request();
      const rawUrl = response.url();
      const contentType = response.headers()["content-type"] || "";
      const entry = {
        method: request.method(),
        status: response.status(),
        content_type: contentType,
        url: sanitizeUrl(rawUrl),
      };
      networkLog.push(entry);

      let url;
      try {
        url = new URL(rawUrl);
      } catch {
        return;
      }
      if (
        !allowedBodyHosts.has(url.hostname) ||
        !interestingUrl.test(rawUrl) ||
        !structuredContent.test(contentType)
      ) {
        return;
      }

      const body = await response.body().catch(() => null);
      if (!body || body.length > 2_000_000) {
        return;
      }
      const decoded = body.toString("utf8");
      if (!/A148/i.test(decoded) && !locationField.test(decoded)) {
        return;
      }

      responseIndex += 1;
      const fileName = safeFileName(responseIndex, rawUrl, contentType);
      await fs.writeFile(path.join(outputDir, "public_responses", fileName), decoded, "utf8");
      savedResponses.push({
        file: `public_responses/${fileName}`,
        bytes: body.length,
        contains_a148: /A148/i.test(decoded),
        contains_location_fields: locationField.test(decoded),
        url: entry.url,
      });
    })();
    responseTasks.push(task);
  });

  const navigation = await page.goto(START_URL, {
    waitUntil: "domcontentloaded",
    timeout: 45_000,
  });
  await page.waitForTimeout(4_000);
  initialText = await savePageState(page, "01_home");

  if (!blockedText.test(initialText)) {
    let searchInput = await firstVisible(
      page.locator([
        'input[type="search"]',
        'input[placeholder*="\ub178\uc120"]',
        'input[placeholder*="\uac80\uc0c9"]',
        'input[title*="\ub178\uc120"]',
        'input[title*="\uac80\uc0c9"]',
        'input[aria-label*="\ub178\uc120"]',
        'input[aria-label*="\uac80\uc0c9"]',
      ].join(",")),
    );

    if (!searchInput) {
      const searchEntry = await firstVisible(
        page.locator('a:has-text("\ub178\uc120\uac80\uc0c9"), button:has-text("\ub178\uc120\uac80\uc0c9"), a:has-text("\ubc84\uc2a4\uac80\uc0c9"), button:has-text("\ubc84\uc2a4\uac80\uc0c9")'),
      );
      if (searchEntry) {
        await searchEntry.click();
        await page.waitForTimeout(2_000);
        searchInput = await firstVisible(
          page.locator('input[type="search"], input[type="text"], input:not([type])'),
        );
      }
    }

    if (searchInput) {
      await searchInput.fill(QUERY);
      await searchInput.press("Enter");
      searchAttempt = "submitted";
      await page.waitForTimeout(5_000);
    } else {
      searchAttempt = "no_visible_search_input";
    }
  } else {
    searchAttempt = "blocked_before_search";
  }

  searchText = await savePageState(page, "02_after_search");

  if (!blockedText.test(searchText)) {
    const routeResult = await firstVisible(
      page.locator('a:has-text("A148"), button:has-text("A148"), [role="button"]:has-text("A148")'),
    );
    if (routeResult) {
      await routeResult.click();
      routeClick = "clicked";
      await page.waitForTimeout(5_000);
    } else {
      routeClick = "no_clickable_a148_result";
    }
  } else {
    routeClick = "blocked_after_search";
  }

  routeText = await savePageState(page, "03_route_result");
  await Promise.allSettled(responseTasks);

  await fs.writeFile(
    path.join(outputDir, "network_metadata.json"),
    `${JSON.stringify(networkLog, null, 2)}\n`,
    "utf8",
  );

  const finalText = `${initialText}\n${searchText}\n${routeText}`;
  const responseEvidence = savedResponses.some(
    (item) => item.contains_a148 && item.contains_location_fields,
  );
  const pageHasA148 = /A148/i.test(finalText);
  const pageBlocked = blockedText.test(finalText);
  const locationEvidence = locationField.test(finalText) || savedResponses.some(
    (item) => item.contains_location_fields,
  );

  const summary = {
    tested_at_utc: new Date().toISOString(),
    start_url: START_URL,
    initial_http_status: navigation?.status() ?? null,
    final_url: sanitizeUrl(page.url()),
    page_title: await page.title(),
    search_attempt: searchAttempt,
    route_click: routeClick,
    blocked: pageBlocked,
    a148_visible_or_returned: pageHasA148 || savedResponses.some((item) => item.contains_a148),
    location_fields_visible_or_returned: locationEvidence,
    public_response_with_a148_and_location_fields: responseEvidence,
    saved_public_responses: savedResponses,
    verdict: pageBlocked
      ? "blocked"
      : responseEvidence
        ? "feasible_for_position_sampling"
        : pageHasA148
          ? "route_visible_but_no_machine_readable_position_evidence"
          : "a148_not_found",
    privacy_note: "No cookies, storage, request/response headers, credentials, or full HAR were saved.",
  };

  await fs.writeFile(
    path.join(outputDir, "summary.json"),
    `${JSON.stringify(summary, null, 2)}\n`,
    "utf8",
  );
  await fs.writeFile(
    path.join(outputDir, "summary.md"),
    [
      "## A148 no-key public-page probe",
      "",
      `- Verdict: **${summary.verdict}**`,
      `- Initial HTTP status: ${summary.initial_http_status ?? "unknown"}`,
      `- Final URL: ${summary.final_url}`,
      `- A148 visible/returned: ${summary.a148_visible_or_returned}`,
      `- Location fields visible/returned: ${summary.location_fields_visible_or_returned}`,
      `- Saved relevant public responses: ${summary.saved_public_responses.length}`,
      "- Safety: no login, key, CAPTCHA bypass, cookies, storage, headers, or full HAR.",
      "",
    ].join("\n"),
    "utf8",
  );
} catch (error) {
  fatalError = error;
  await fs.writeFile(
    path.join(outputDir, "fatal_error.txt"),
    `${error.stack || error.message}\n`,
    "utf8",
  );
  const summary = {
    tested_at_utc: new Date().toISOString(),
    start_url: START_URL,
    verdict: "probe_error",
    error: error.message,
    privacy_note: "No cookies, storage, request/response headers, credentials, or full HAR were saved.",
  };
  await fs.writeFile(
    path.join(outputDir, "summary.json"),
    `${JSON.stringify(summary, null, 2)}\n`,
    "utf8",
  );
  await fs.writeFile(
    path.join(outputDir, "summary.md"),
    `## A148 no-key public-page probe\n\n- Verdict: **probe_error**\n- Error: ${error.message}\n`,
    "utf8",
  );
} finally {
  await browser?.close().catch(() => {});
}

if (fatalError) {
  process.exitCode = 1;
}
