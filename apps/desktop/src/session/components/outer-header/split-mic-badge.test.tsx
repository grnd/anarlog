import { cleanup, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SplitMicBadge } from "./split-mic-badge";

const mocks = vi.hoisted(() => ({
  mode: "active",
  splitStereoMic: true,
}));

vi.mock("@lingui/react/macro", () => ({
  Trans: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@anlg/ui/components/ui/tooltip", () => ({
  Tooltip: ({ children }: { children: ReactNode }) => <>{children}</>,
  TooltipTrigger: ({ children }: { children: ReactNode }) => <>{children}</>,
  TooltipContent: () => null,
}));

vi.mock("~/stt/contexts", () => ({
  useListener: (selector: (state: unknown) => unknown) =>
    selector({
      getSessionMode: () => mocks.mode,
      live: { splitStereoMic: mocks.splitStereoMic },
    }),
}));

describe("SplitMicBadge", () => {
  afterEach(() => {
    cleanup();
    mocks.mode = "active";
    mocks.splitStereoMic = true;
  });

  it("shows for a running split capture", () => {
    render(<SplitMicBadge sessionId="s1" />);
    expect(screen.getByText("Split mic")).toBeTruthy();
  });

  it("is hidden for a normal capture", () => {
    mocks.splitStereoMic = false;
    render(<SplitMicBadge sessionId="s1" />);
    expect(screen.queryByText("Split mic")).toBeNull();
  });

  it("is hidden when the session is not recording", () => {
    mocks.mode = "inactive";
    render(<SplitMicBadge sessionId="s1" />);
    expect(screen.queryByText("Split mic")).toBeNull();
  });
});
