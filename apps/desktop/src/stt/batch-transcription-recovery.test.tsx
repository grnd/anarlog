import { render, waitFor } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";

import type { TranscriptionSession } from "@anlg/plugin-transcription";

const mocks = vi.hoisted(() => ({
  recoverRunningBatchSessions: vi.fn(),
  runBatch: vi.fn(),
  useConfigValue: vi.fn(),
}));

vi.mock("~/services/enhancer", () => ({
  getEnhancerService: () => null,
}));

vi.mock("~/session/queries", () => ({
  useSession: (sessionId: string) => ({ id: sessionId }),
}));

vi.mock("~/shared/config", () => ({
  useConfigValue: mocks.useConfigValue,
}));

vi.mock("~/store/zustand/listener/general-batch", () => ({
  recoverRunningBatchSessions: mocks.recoverRunningBatchSessions,
}));

vi.mock("~/store/zustand/listener/instance", () => ({
  listenerStore: { getState: () => ({ handleBatchFailed: vi.fn() }) },
}));

vi.mock("~/stt/useRunBatch", () => ({
  isStoppedTranscriptionError: () => false,
  useRunBatch: () => mocks.runBatch,
}));

import { BatchTranscriptionRecovery } from "./batch-transcription-recovery";

beforeEach(() => {
  vi.clearAllMocks();
  mocks.runBatch.mockResolvedValue(undefined);
});

test.each([
  {
    name: "an interrupted upload stays unfiltered with the setting on",
    resumeContext: '{"promotion":"whole_session"}',
    setting: true,
    splitStereoMic: false,
  },
  {
    name: "a split batch keeps filtering after the setting was turned off",
    resumeContext: '{"promotion":"whole_session","splitStereoMic":true}',
    setting: false,
    splitStereoMic: true,
  },
])(
  "resumes with the split mode the batch started with: $name",
  async ({ resumeContext, setting, splitStereoMic }) => {
    mocks.useConfigValue.mockImplementation((key: string) =>
      key === "microphone_split_stereo" ? setting : undefined,
    );
    const session: TranscriptionSession = {
      session_id: "session-1",
      file_path: "/tmp/session.wav",
      provider: "deepgram",
      model: "nova-3",
      started_at_ms: 1_000,
      resume_context: resumeContext,
      completed: false,
    };
    mocks.recoverRunningBatchSessions.mockImplementation(
      async (
        _get: unknown,
        onResumable: (s: TranscriptionSession[]) => void,
      ) => {
        onResumable([session]);
        return () => {};
      },
    );

    render(<BatchTranscriptionRecovery />);

    await waitFor(() => expect(mocks.runBatch).toHaveBeenCalled());
    expect(mocks.runBatch.mock.calls[0]?.[1]?.splitStereoMic === true).toBe(
      splitStereoMic,
    );
  },
);
