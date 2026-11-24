import { Trans } from "@lingui/react/macro";

import { Badge } from "@anlg/ui/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@anlg/ui/components/ui/tooltip";

import { useListener } from "~/stt/contexts";

export function SplitMicBadge({ sessionId }: { sessionId: string }) {
  const splitActive = useListener(
    (state) =>
      state.getSessionMode(sessionId) === "active" && state.live.splitStereoMic,
  );
  if (!splitActive) return null;

  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild>
        <Badge variant="secondary" size="sm" className="mr-1 shrink-0">
          <Trans>Split mic</Trans>
        </Badge>
      </TooltipTrigger>
      <TooltipContent side="bottom">
        <Trans>
          Left and right channels are two speakers. System audio is not
          recorded.
        </Trans>
      </TooltipContent>
    </Tooltip>
  );
}
