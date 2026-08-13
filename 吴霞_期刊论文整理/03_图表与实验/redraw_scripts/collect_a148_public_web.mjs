#!/usr/bin/env node
/**
 * Collect A148 positions from Seoul's public bus-information webpage.
 *
 * The collector opens https://bus.go.kr/ in an ordinary headless browser and
 * calls only the same-origin JSON resources used by that public page. It does
 * not log in, use an API key, bypass a challenge, or retain browser storage,
 * cookies, headers, or a HAR file.
 */

import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const START_URL = "https://bus.go.kr/";
const ROUTE_NUMBER = "A148";
const EXPECTED_ROUTE_ID = "101000009";
const START_SECTION_MAX = 2;
const END_SECTION = 41;
const DEFAULT_OUTPUT_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../source_data/a148_realtime",
);

const CSV_FIELDS = [
  "api_received_at_utc",
  "api_received_at_seoul",
  "dataTm",
  "vehId",
  "plainNo",
  "routeId",
  "sectOrd",
  "sectionId",
  "sectDist",
  "fullSectDist",
  "stopFlag",
  "lastStnId",
  "tmX",
  "tmY",
  "posX",
  "posY",
  "busType",
  "congetion",
  "islastyn",
  "trnstnid",
  "rtDist",
  "lastStTm",
  "nextStTm",
];

function parseArgs(argv) {
  const args = {
    interval: 15,
    durationMinutes: 230,
    idleStopMinutes: 15,
    timeoutSeconds: 25,
    outputDir: DEFAULT_OUTPUT_DIR,
    once: false,
    selfTest: false,
  };
  const valueOptions = new Map([
    ["--interval", "interval"],
    ["--duration-minutes", "durationMinutes"],
    ["--idle-stop-minutes", "idleStopMinutes"],
    ["--timeout", "timeoutSeconds"],
    ["--output-dir", "outputDir"],
  ]);
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === "--once") {
      args.once = true;
      continue;
    }
    if (token === "--self-test") {
      args.selfTest = true;
      continue;
    }
    const property = valueOptions.get(token);
    if (!property || index + 1 >= argv.length) {
      throw new Error(`Unknown or incomplete option: ${token}`);
    }
    const rawValue = argv[index + 1];
    index += 1;
    if (property === "outputDir") {
      args.outputDir = path.resolve(rawValue);
    } else {
      const number = Number(rawValue);
      if (!Number.isFinite(number) || number <= 0) {
        throw new Error(`${token} must be a positive number.`);
      }
      args[property] = number;
    }
  }
  if (!args.once && args.interval < 10) {
    throw new Error("--interval must be at least 10 seconds for continuous collection.");
  }
  return args;
}

function seoulParts(date = new Date()) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Seoul",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
  }).formatToParts(date);
  return Object.fromEntries(parts.map(({ type, value }) => [type, value]));
}

function serviceDay(date = new Date()) {
  const parts = seoulParts(date);
  return `${parts.year}${parts.month}${parts.day}`;
}

function seoulIso(date = new Date()) {
  const parts = seoulParts(date);
  return `${parts.year}-${parts.month}-${parts.day}T${parts.hour}:${parts.minute}:${parts.second}+09:00`;
}

function compactTimestamp(rawValue, fallbackDate) {
  const text = String(rawValue ?? "").trim();
  const compact = text.match(/^(\d{4})-?(\d{2})-?(\d{2})[T ]?(\d{2}):?(\d{2}):?(\d{2})/);
  if (compact) {
    return compact.slice(1).join("");
  }
  const parts = seoulParts(fallbackDate);
  return `${parts.year}${parts.month}${parts.day}${parts.hour}${parts.minute}${parts.second}`;
}

function firstValue(record, names, fallback = "") {
  for (const name of names) {
    const value = record?.[name];
    if (value !== undefined && value !== null && String(value).trim() !== "") {
      return value;
    }
  }
  return fallback;
}

function normalizeVehicle(record, receivedAt, responseTimestamp, routeId) {
  const longitude = firstValue(record, ["posX", "gpsX", "tmX", "longitude", "lng"]);
  const latitude = firstValue(record, ["posY", "gpsY", "tmY", "latitude", "lat"]);
  const vehicleId = firstValue(
    record,
    ["vehId", "vehid", "vehicleId", "busId", "plainNo"],
    "unknown_vehicle",
  );
  const row = {
    api_received_at_utc: receivedAt.toISOString(),
    api_received_at_seoul: seoulIso(receivedAt),
    dataTm: compactTimestamp(
      firstValue(
        record,
        ["dataTm", "dataTime", "collectTm", "lastUpdate"],
        responseTimestamp,
      ),
      receivedAt,
    ),
    vehId: vehicleId,
    plainNo: firstValue(record, ["plainNo", "plateNo", "busNo"]),
    routeId: firstValue(record, ["routeId", "busRouteId", "rtId"], routeId),
    sectOrd: firstValue(record, ["sectOrd", "sectord", "sectionSeq", "seq", "ord"]),
    sectionId: firstValue(record, ["sectionId", "sectId", "section"]),
    sectDist: firstValue(record, ["sectDist", "sectionDistance"]),
    fullSectDist: firstValue(record, ["fullSectDist", "fullSectionDistance"]),
    stopFlag: firstValue(record, ["stopFlag", "stopflag", "isStop"], 0),
    lastStnId: firstValue(record, ["lastStnId", "lastStationId", "stationId"]),
    tmX: longitude,
    tmY: latitude,
    posX: longitude,
    posY: latitude,
    busType: firstValue(record, ["busType", "busType1"]),
    congetion: firstValue(record, ["congetion", "congestion"]),
    islastyn: firstValue(record, ["islastyn", "isLastYn"]),
    trnstnid: firstValue(record, ["trnstnid", "turnStationId"]),
    rtDist: firstValue(record, ["rtDist", "routeDistance"]),
    lastStTm: firstValue(record, ["lastStTm", "lastStationTime"]),
    nextStTm: firstValue(record, ["nextStTm", "nextStationTime"]),
  };
  return Object.fromEntries(CSV_FIELDS.map((field) => [field, row[field] ?? ""]));
}

function observeVehicleCoverage(coverageByVehicle, rows, seenAtMs) {
  for (const row of rows) {
    const vehicleId = String(row.vehId || "").trim();
    const section = Number(row.sectOrd);
    if (!vehicleId || !Number.isFinite(section)) continue;
    const coverage = coverageByVehicle.get(vehicleId) || {
      minSection: section,
      maxSection: section,
      sampleCount: 0,
      lastSeenAtMs: seenAtMs,
    };
    coverage.minSection = Math.min(coverage.minSection, section);
    coverage.maxSection = Math.max(coverage.maxSection, section);
    coverage.sampleCount += 1;
    coverage.lastSeenAtMs = seenAtMs;
    coverageByVehicle.set(vehicleId, coverage);
  }
}

function findCompleteVehicle(coverageByVehicle) {
  for (const [vehicleId, coverage] of coverageByVehicle) {
    if (
      coverage.minSection <= START_SECTION_MAX &&
      coverage.maxSection >= END_SECTION - 1
    ) {
      return vehicleId;
    }
  }
  return "";
}

function summarizeVehicleCoverage(coverageByVehicle) {
  return Object.fromEntries(
    [...coverageByVehicle.entries()].map(([vehicleId, coverage]) => [
      vehicleId,
      {
        min_section: coverage.minSection,
        max_section: coverage.maxSection,
        samples_observed: coverage.sampleCount,
      },
    ]),
  );
}

function parsePositionPayload(payload) {
  const response = payload?.ResponseVO;
  if (!response || Number(response.code) !== 0) {
    throw new Error(`Public position response failed: ${response?.message || "missing ResponseVO"}`);
  }
  const data = response.data;
  if (!data || !Array.isArray(data.resultRouteBuspos)) {
    throw new Error("Public position response does not contain resultRouteBuspos.");
  }
  return {
    responseTimestamp: response.timestamp || "",
    message: response.message || "",
    vehicles: data.resultRouteBuspos,
    routePath: Array.isArray(data.resultRoutePath) ? data.resultRoutePath : [],
    routeStops: Array.isArray(data.resultRouteStop) ? data.resultRouteStop : [],
    rawData: data,
  };
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

async function appendJsonLine(filePath, value) {
  await fs.appendFile(filePath, `${JSON.stringify(value)}\n`, "utf8");
}

async function appendCsvRows(filePath, rows) {
  if (rows.length === 0) return;
  let prefix = "";
  try {
    const stat = await fs.stat(filePath);
    if (stat.size === 0) prefix = `${CSV_FIELDS.join(",")}\n`;
  } catch {
    prefix = `${CSV_FIELDS.join(",")}\n`;
  }
  const lines = rows.map((row) => CSV_FIELDS.map((field) => csvCell(row[field])).join(","));
  await fs.appendFile(filePath, `${prefix}${lines.join("\n")}\n`, "utf8");
}

async function fetchPageJson(page, endpoint, parameters, timeoutSeconds) {
  return page.evaluate(
    async ({ endpointPath, query, timeoutMs }) => {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const url = new URL(endpointPath, window.location.origin);
        for (const [name, value] of Object.entries(query)) {
          url.searchParams.set(name, String(value));
        }
        url.searchParams.set("_dc", String(Date.now()));
        const response = await fetch(url, {
          method: "GET",
          credentials: "same-origin",
          headers: { Accept: "application/json" },
          signal: controller.signal,
        });
        const text = await response.text();
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        return JSON.parse(text);
      } finally {
        clearTimeout(timer);
      }
    },
    {
      endpointPath: endpoint,
      query: parameters,
      timeoutMs: timeoutSeconds * 1000,
    },
  );
}

async function discoverRoute(page, timeoutSeconds) {
  const payload = await fetchPageJson(
    page,
    "/sbus/bus/selectVApiTotalstr.do",
    {
      pageIndex: 1,
      recordCountPerPage: 50,
      strkey: ROUTE_NUMBER,
      strdiv: 1,
      strid: "",
    },
    timeoutSeconds,
  );
  const response = payload?.ResponseVO;
  const routes = response?.data?.resultList;
  if (Number(response?.code) !== 0 || !Array.isArray(routes)) {
    throw new Error(`A148 route search failed: ${response?.message || "unexpected response"}`);
  }
  const route = routes.find(
    (item) => String(item.strno || "").toUpperCase() === ROUTE_NUMBER,
  );
  if (!route) {
    throw new Error("The public webpage did not return an exact A148 route result.");
  }
  const routeId = String(route.strid || "");
  if (routeId !== EXPECTED_ROUTE_ID) {
    throw new Error(`A148 route ID changed from ${EXPECTED_ROUTE_ID} to ${routeId || "empty"}.`);
  }
  return { route, searchTimestamp: response.timestamp || "" };
}

function buildPaths(outputDir, day) {
  return {
    csv: path.join(outputDir, `a148_raw_positions_${day}.csv`),
    raw: path.join(outputDir, `a148_public_vehicle_snapshots_${day}.jsonl`),
    log: path.join(outputDir, `a148_collection_log_${day}.jsonl`),
    route: path.join(outputDir, `a148_public_route_snapshot_${day}.json`),
    screenshot: path.join(outputDir, `a148_public_page_${day}.png`),
    summaryJson: path.join(outputDir, `a148_collection_summary_${day}.json`),
    summaryMd: path.join(outputDir, `a148_collection_summary_${day}.md`),
  };
}

async function writeSummary(paths, summary) {
  await fs.writeFile(paths.summaryJson, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  const markdown = [
    "## A148 public-web capture",
    "",
    `- Verdict: **${summary.verdict}**`,
    `- Route ID: ${summary.route_id}`,
    `- Requests / successful: ${summary.requests} / ${summary.successful_requests}`,
    `- Raw vehicle records / unique samples: ${summary.raw_vehicle_records} / ${summary.unique_samples}`,
    `- Static route points / stops: ${summary.route_path_points} / ${summary.route_stops}`,
    `- Start / final section observed: ${summary.reached_start_section} / ${summary.reached_final_section}`,
    `- Complete vehicle trajectory: ${summary.complete_vehicle_id || "none"}`,
    `- Per-vehicle section ranges: ${JSON.stringify(summary.vehicle_section_ranges)}`,
    `- Started / finished (Seoul): ${summary.started_at_seoul} / ${summary.finished_at_seoul}`,
    "- Source: Seoul public bus-information webpage; no account or API key used.",
    "- Privacy: no cookies, storage, headers, credentials, or HAR were retained.",
    "",
  ].join("\n");
  await fs.writeFile(paths.summaryMd, markdown, "utf8");
}

function runSelfTest() {
  const receivedAt = new Date("2026-08-11T18:30:15Z");
  const parsed = parsePositionPayload({
    ResponseVO: {
      code: 0,
      timestamp: "2026-08-12T03:30:15.000",
      data: {
        resultRoutePath: [{ ord: 1 }],
        resultRouteStop: [{ seq: 1 }],
        resultRouteBuspos: [{
          vehId: "test-vehicle",
          plainNo: "test-plate",
          sectOrd: 1,
          posX: 127.073782,
          posY: 37.661233,
          stopFlag: 1,
        }],
      },
    },
  });
  const row = normalizeVehicle(
    parsed.vehicles[0],
    receivedAt,
    parsed.responseTimestamp,
    EXPECTED_ROUTE_ID,
  );
  if (
    parsed.routePath.length !== 1 ||
    parsed.routeStops.length !== 1 ||
    row.dataTm !== "20260812033015" ||
    String(row.sectOrd) !== "1" ||
    Number(row.tmX) !== 127.073782 ||
    row.routeId !== EXPECTED_ROUTE_ID
  ) {
    throw new Error("Self-test failed.");
  }
  const coverage = new Map();
  observeVehicleCoverage(coverage, [row], receivedAt.getTime());
  observeVehicleCoverage(
    coverage,
    [{ ...row, sectOrd: END_SECTION - 1 }],
    receivedAt.getTime() + 1000,
  );
  if (findCompleteVehicle(coverage) !== "test-vehicle") {
    throw new Error("Coverage self-test failed.");
  }
  const partialCoverage = new Map();
  observeVehicleCoverage(
    partialCoverage,
    [{ ...row, sectOrd: 15 }, { ...row, sectOrd: END_SECTION - 1 }],
    receivedAt.getTime(),
  );
  if (findCompleteVehicle(partialCoverage) !== "") {
    throw new Error("Partial-coverage self-test failed.");
  }
  process.stdout.write("Collector self-test passed.\n");
}

async function collect(args) {
  const { chromium } = await import("playwright");
  await fs.mkdir(args.outputDir, { recursive: true });
  const day = serviceDay();
  const paths = buildPaths(args.outputDir, day);
  const startedAt = new Date();
  const summary = {
    verdict: "running",
    route_id: EXPECTED_ROUTE_ID,
    started_at_utc: startedAt.toISOString(),
    started_at_seoul: seoulIso(startedAt),
    finished_at_utc: null,
    finished_at_seoul: null,
    requests: 0,
    successful_requests: 0,
    request_errors: 0,
    raw_vehicle_records: 0,
    unique_samples: 0,
    route_path_points: 0,
    route_stops: 0,
    reached_start_section: false,
    reached_final_section: false,
    complete_vehicle_id: "",
    vehicle_section_ranges: {},
  };
  let browser;
  let seenVehicle = false;
  let completeVehicleId = "";
  let routeSnapshotSaved = false;
  const uniqueSamples = new Set();
  const coverageByVehicle = new Map();

  try {
    browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({
      locale: "ko-KR",
      timezoneId: "Asia/Seoul",
      viewport: { width: 1440, height: 1000 },
    });
    const page = await context.newPage();
    const navigation = await page.goto(START_URL, {
      waitUntil: "domcontentloaded",
      timeout: args.timeoutSeconds * 1000,
    });
    if (!navigation || navigation.status() !== 200) {
      throw new Error(`The public webpage returned HTTP ${navigation?.status() ?? "unknown"}.`);
    }
    await page.waitForTimeout(2500);
    const visibleText = await page.locator("body").innerText({ timeout: 10_000 });
    if (/captcha|access denied|forbidden|unusual traffic|robot|\ub85c\ubd07|\uc811\uadfc\uc774 \ucc28\ub2e8/i.test(visibleText)) {
      throw new Error("The public webpage blocked automated access; collection stopped without bypassing it.");
    }
    await page.screenshot({ path: paths.screenshot, fullPage: true });

    const { route, searchTimestamp } = await discoverRoute(page, args.timeoutSeconds);
    const deadline = Date.now() + (args.once ? 60_000 : args.durationMinutes * 60_000);

    while (Date.now() < deadline) {
      const cycleStarted = Date.now();
      const receivedAt = new Date();
      summary.requests += 1;
      let status = "ok";
      let message = "";
      let vehicles = [];
      let normalizedRows = [];
      let newRows = [];
      let responseTimestamp = "";

      try {
        const payload = await fetchPageJson(
          page,
          "/sbus/bus/selectBusposInfo.do",
          { routeId: EXPECTED_ROUTE_ID, isLowBus: "N" },
          args.timeoutSeconds,
        );
        const parsed = parsePositionPayload(payload);
        vehicles = parsed.vehicles;
        responseTimestamp = parsed.responseTimestamp;
        summary.successful_requests += 1;
        summary.raw_vehicle_records += vehicles.length;
        summary.route_path_points = Math.max(summary.route_path_points, parsed.routePath.length);
        summary.route_stops = Math.max(summary.route_stops, parsed.routeStops.length);

        if (!routeSnapshotSaved) {
          await fs.writeFile(
            paths.route,
            `${JSON.stringify({
              captured_at_utc: receivedAt.toISOString(),
              search_timestamp: searchTimestamp,
              route,
              response_timestamp: responseTimestamp,
              route_path: parsed.routePath,
              route_stops: parsed.routeStops,
            }, null, 2)}\n`,
            "utf8",
          );
          routeSnapshotSaved = true;
        }

        await appendJsonLine(paths.raw, {
          received_at_utc: receivedAt.toISOString(),
          received_at_seoul: seoulIso(receivedAt),
          request_number: summary.requests,
          response_timestamp: responseTimestamp,
          route_id: EXPECTED_ROUTE_ID,
          vehicles,
        });

        normalizedRows = vehicles.map((vehicle) => normalizeVehicle(
            vehicle,
            receivedAt,
            responseTimestamp,
            EXPECTED_ROUTE_ID,
          ));
        observeVehicleCoverage(coverageByVehicle, normalizedRows, Date.now());
        completeVehicleId ||= findCompleteVehicle(coverageByVehicle);
        newRows = normalizedRows.filter((row) => {
            const identity = `${row.vehId}|${row.dataTm}`;
            if (uniqueSamples.has(identity)) return false;
            uniqueSamples.add(identity);
            return true;
          });
        await appendCsvRows(paths.csv, newRows);

        if (vehicles.length > 0) {
          seenVehicle = true;
        }
      } catch (error) {
        status = "request_error";
        message = `${error.name}: ${error.message}`;
        summary.request_errors += 1;
      }

      summary.unique_samples = uniqueSamples.size;
      summary.reached_start_section = [...coverageByVehicle.values()].some(
        (coverage) => coverage.minSection <= START_SECTION_MAX,
      );
      summary.reached_final_section = [...coverageByVehicle.values()].some(
        (coverage) => coverage.maxSection >= END_SECTION - 1,
      );
      summary.complete_vehicle_id = completeVehicleId;
      summary.vehicle_section_ranges = summarizeVehicleCoverage(coverageByVehicle);
      await appendJsonLine(paths.log, {
        received_at_utc: receivedAt.toISOString(),
        received_at_seoul: seoulIso(receivedAt),
        request_number: summary.requests,
        status,
        message,
        response_timestamp: responseTimestamp,
        vehicles_returned: vehicles.length,
        new_samples: newRows.length,
      });
      process.stdout.write(
        `${seoulIso(receivedAt)} request=${summary.requests} ` +
        `vehicles=${vehicles.length} new_samples=${newRows.length} status=${status}\n`,
      );

      if (args.once) break;
      if (
        completeVehicleId &&
        Date.now() - coverageByVehicle.get(completeVehicleId).lastSeenAtMs >=
          args.idleStopMinutes * 60_000
      ) {
        process.stdout.write("Round trip appears complete; stopping after the idle period.\n");
        break;
      }
      const sleepMs = Math.max(0, args.interval * 1000 - (Date.now() - cycleStarted));
      await new Promise((resolve) => setTimeout(resolve, sleepMs));
    }

    summary.verdict = args.once
      ? summary.successful_requests > 0
        ? "public_endpoint_accessible"
        : "public_endpoint_failed"
      : !seenVehicle
        ? "no_a148_vehicle_observed"
        : completeVehicleId
          ? "vehicle_trajectory_collected"
          : "partial_vehicle_trajectory";
  } catch (error) {
    summary.verdict = "collector_error";
    summary.fatal_error = `${error.name}: ${error.message}`;
  } finally {
    await browser?.close().catch(() => {});
    const finishedAt = new Date();
    summary.finished_at_utc = finishedAt.toISOString();
    summary.finished_at_seoul = seoulIso(finishedAt);
    await writeSummary(paths, summary);
  }

  process.stdout.write(`${JSON.stringify(summary)}\n`);
  if (summary.verdict === "collector_error" || summary.verdict === "public_endpoint_failed") {
    return 2;
  }
  if (!args.once && summary.verdict === "no_a148_vehicle_observed") return 3;
  if (!args.once && summary.verdict === "partial_vehicle_trajectory") return 4;
  return 0;
}

async function main() {
  let args;
  try {
    args = parseArgs(process.argv.slice(2));
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    return 64;
  }
  if (args.selfTest) {
    runSelfTest();
    return 0;
  }
  return collect(args);
}

process.exitCode = await main();
