import type { Claim, Evidence, Verdict } from "@/lib/types";

type ClaimCardProps = {
  index: number;
  claim: Claim;
  verdict?: Verdict;
  evidence: Evidence[];
};

const verdictStyles = {
  supported: {
    label: "Supported",
    badge: "border-[#277453]/30 bg-[#dff0e6] text-[#18543b]",
    bar: "bg-[#277453]",
  },
  contradicted: {
    label: "Contradicted",
    badge: "border-[#b43b25]/30 bg-[#f7dfd8] text-[#942d1a]",
    bar: "bg-[#d54b2a]",
  },
  insufficient_evidence: {
    label: "Insufficient evidence",
    badge: "border-[#5f6963]/25 bg-[#e8e8e2] text-[#4f5953]",
    bar: "bg-[#7c8580]",
  },
} as const;

export default function ClaimCard({
  index,
  claim,
  verdict,
  evidence,
}: ClaimCardProps) {
  const label = verdict?.label ?? "insufficient_evidence";
  const style = verdictStyles[label];
  const confidence = Math.round((verdict?.confidence ?? 0) * 100);

  return (
    <article className="overflow-hidden border border-[#17201c]/20 bg-[#fcfaf4]">
      <div className={`h-1.5 ${style.bar}`} />
      <div className="grid gap-7 p-5 sm:p-8 lg:grid-cols-[minmax(0,1fr)_12rem] lg:p-10">
        <div>
          <div className="mb-5 flex flex-wrap items-center gap-3">
            <span className="font-mono text-xs font-semibold text-[#78827c]">
              CLAIM {String(index).padStart(2, "0")}
            </span>
            <span
              className={`rounded-full border px-3 py-1 text-xs font-bold tracking-wide ${style.badge}`}
            >
              {style.label}
            </span>
          </div>
          <h3 className="max-w-3xl text-2xl leading-tight font-semibold tracking-[-0.025em] sm:text-3xl">
            {claim.text}
          </h3>
          <p className="mt-5 max-w-3xl text-sm leading-6 text-[#59655e] sm:text-base sm:leading-7">
            {verdict?.explanation ??
              "Axiom could not establish a reliable verdict for this claim."}
          </p>
        </div>

        <div className="border-t border-[#17201c]/15 pt-5 lg:border-t-0 lg:border-l lg:pt-0 lg:pl-8">
          <p className="text-xs font-semibold tracking-[0.16em] text-[#707a74] uppercase">
            Confidence
          </p>
          <p className="mt-2 text-5xl font-semibold tracking-[-0.06em] tabular-nums">
            {confidence}
            <span className="ml-1 text-xl text-[#6e7872]">%</span>
          </p>
          <div className="mt-4 h-1.5 overflow-hidden bg-[#deddd5]">
            <div
              className={`h-full ${style.bar}`}
              style={{ width: `${confidence}%` }}
            />
          </div>
        </div>
      </div>

      <details className="group border-t border-[#17201c]/15 bg-white/55">
        <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-5 py-4 text-sm font-semibold transition hover:bg-white sm:px-8 lg:px-10 [&::-webkit-details-marker]:hidden">
          <span>
            View {evidence.length} source{evidence.length === 1 ? "" : "s"}
          </span>
          <span
            aria-hidden="true"
            className="text-xl transition-transform group-open:rotate-45"
          >
            +
          </span>
        </summary>

        <div className="grid gap-px border-t border-[#17201c]/10 bg-[#17201c]/10">
          {evidence.length ? (
            evidence.map((item, evidenceIndex) => (
              <EvidenceRow
                key={item.id}
                evidence={item}
                index={evidenceIndex + 1}
              />
            ))
          ) : (
            <p className="bg-[#fcfaf4] px-5 py-6 text-sm text-[#68736c] sm:px-8 lg:px-10">
              No evidence met Axiom&apos;s relevance threshold.
            </p>
          )}
        </div>
      </details>
    </article>
  );
}

function EvidenceRow({
  evidence,
  index,
}: {
  evidence: Evidence;
  index: number;
}) {
  const relevance = Math.round(
    (evidence.combined_score ?? evidence.reranker_score ?? evidence.score) * 100,
  );

  return (
    <div className="grid gap-4 bg-[#fcfaf4] px-5 py-6 sm:grid-cols-[2rem_minmax(0,1fr)_auto] sm:px-8 lg:px-10">
      <span className="font-mono text-xs text-[#838c87]">
        {String(index).padStart(2, "0")}
      </span>
      <div>
        <p className="text-sm leading-6 text-[#39453f]">{evidence.text}</p>
        <a
          href={evidence.source_url}
          target="_blank"
          rel="noreferrer"
          className="mt-3 inline-flex items-center gap-2 text-sm font-semibold text-[#a7351f] underline decoration-[#a7351f]/30 underline-offset-4 hover:decoration-[#a7351f] focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#d54b2a]"
        >
          {evidence.source_name}
          <span aria-hidden="true">↗</span>
        </a>
      </div>
      <span className="h-fit w-fit rounded-full bg-[#e8e5dc] px-2.5 py-1 font-mono text-[0.65rem] font-semibold text-[#5d6761] uppercase">
        {relevance}% match
      </span>
    </div>
  );
}
