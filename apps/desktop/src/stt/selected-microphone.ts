import { invoke } from "@tauri-apps/api/core";

type InputDevice = { name: string; is_default: boolean };

// The name of the microphone a recording will actually open: the explicit
// choice when it is connected, or the current system default input when the
// setting is empty or the chosen mic is unplugged. Empty when unknown.
export async function resolveSelectedMicrophoneName(
  microphoneDevice: string | null | undefined,
): Promise<string> {
  const explicit = microphoneDevice?.trim() ?? "";
  try {
    const devices = await invoke<InputDevice[]>(
      "plugin:audio-priority|list_input_devices",
    );
    if (explicit) {
      const connected = devices.some(
        (device) => device.name.trim().toLowerCase() === explicit.toLowerCase(),
      );
      if (connected || devices.length === 0) {
        return microphoneDevice!;
      }
    }
    return devices.find((device) => device.is_default)?.name ?? "";
  } catch {
    return explicit ? microphoneDevice! : "";
  }
}
