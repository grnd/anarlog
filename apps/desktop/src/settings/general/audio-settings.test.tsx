import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AudioSettingsView } from "./audio-settings";

function renderAudioSettings({
  microphoneDevice = {
    value: "",
    devices: ["External Microphone"],
    onChange: vi.fn(),
  },
  rememberSpeakers = {
    value: false,
    onChange: vi.fn(),
  },
  splitStereoMic = {
    value: false,
    onChange: vi.fn(),
  } as { value: boolean; device?: string; onChange: (value: boolean) => void },
} = {}) {
  return render(
    <AudioSettingsView
      audioRetention={{ value: "forever", onChange: vi.fn() }}
      microphoneDevice={microphoneDevice}
      rememberSpeakers={rememberSpeakers}
      splitStereoMic={splitStereoMic}
    />,
  );
}

describe("AudioSettingsView", () => {
  afterEach(cleanup);

  it("shows when the selected microphone is unavailable and will fall back", () => {
    renderAudioSettings({
      microphoneDevice: {
        value: "Disconnected Microphone",
        devices: ["External Microphone"],
        onChange: vi.fn(),
      },
    });

    expect(
      screen.getByRole("combobox", { name: "Microphone" }).textContent,
    ).toContain(
      "Disconnected Microphone (Unavailable — using current default)",
    );
  });

  it.each([
    [false, true],
    [true, false],
  ])("toggles remember speakers from %s to %s", (value, nextValue) => {
    const onChange = vi.fn();
    renderAudioSettings({
      rememberSpeakers: { value, onChange },
    });

    const toggle = screen.getByRole("switch", { name: "Remember speakers" });
    expect(toggle.getAttribute("aria-checked")).toBe(String(value));

    toggle.click();
    expect(onChange).toHaveBeenCalledWith(nextValue);
  });

  it("toggles split stereo microphone on", () => {
    const onChange = vi.fn();
    renderAudioSettings({
      splitStereoMic: { value: false, onChange },
    });

    screen
      .getByRole("switch", {
        name: "Split stereo microphone into two speakers",
      })
      .click();
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it("shows which microphone the split applies to", () => {
    renderAudioSettings({
      splitStereoMic: {
        value: true,
        device: "DJI MIC MINI",
        onChange: vi.fn(),
      },
    });

    expect(screen.getByText("Applies to: DJI MIC MINI")).toBeTruthy();
  });

  it("hides the microphone line when the split is off", () => {
    renderAudioSettings({
      splitStereoMic: {
        value: false,
        device: "DJI MIC MINI",
        onChange: vi.fn(),
      },
    });

    expect(screen.queryByText(/Applies to/)).toBeNull();
  });

  it("asks to toggle again when the split has no stored microphone", () => {
    renderAudioSettings({
      splitStereoMic: { value: true, device: "", onChange: vi.fn() },
    });

    expect(
      screen.getByText("Turn this off and on again to choose the microphone."),
    ).toBeTruthy();
  });
});
