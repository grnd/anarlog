import { beforeEach, describe, expect, it, vi } from "vitest";

import { resolveSelectedMicrophoneName } from "./selected-microphone";

const state = vi.hoisted(() => ({
  devices: [] as { name: string; is_default: boolean }[] | Error,
  calls: 0,
}));
vi.mock("@tauri-apps/api/core", () => ({
  invoke: async () => {
    state.calls += 1;
    if (state.devices instanceof Error) throw state.devices;
    return state.devices;
  },
}));

describe("resolveSelectedMicrophoneName", () => {
  beforeEach(() => {
    state.calls = 0;
  });

  it("returns the explicit device when it is connected", async () => {
    state.devices = [
      { name: "Built-in Mic", is_default: true },
      { name: "DJI MIC MINI", is_default: false },
    ];
    expect(await resolveSelectedMicrophoneName("DJI MIC MINI")).toBe(
      "DJI MIC MINI",
    );
  });

  it("falls back to the system default when an explicit device is unplugged", async () => {
    state.devices = [{ name: "Built-in Mic", is_default: true }];
    expect(await resolveSelectedMicrophoneName("DJI MIC MINI")).toBe(
      "Built-in Mic",
    );
  });

  it("returns the default input name when none is selected", async () => {
    state.devices = [
      { name: "Other", is_default: false },
      { name: "Built-in Mic", is_default: true },
    ];
    expect(await resolveSelectedMicrophoneName("")).toBe("Built-in Mic");
  });

  it("returns an empty name when the lookup fails with no explicit device", async () => {
    state.devices = new Error("nope");
    expect(await resolveSelectedMicrophoneName("")).toBe("");
  });
});
