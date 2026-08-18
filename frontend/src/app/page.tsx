"use client";

import { FormEvent, useState } from "react";

import ClaimCard from "@/components/claim-card";
import type { AnalyzeResponse } from "@/lib/types";

const presets = [
  {
    label: "EV batteries",
    text: "Every electric vehicle battery must be replaced after five years.",
  },
  {
    label: "Earthquakes",
    text: "Earthquake magnitude is measured on a logarithmic scale.",
  },
];

const API_URL = process.env.NEXT_PUBLIC_API_URL;

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return "Something went wrong while analyzing this text.";
}

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const input = text.trim();

    if (!input) {
      setError("Enter at least one factual claim to analyze.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      if (!API_URL) {
        throw new Error("The Axiom API URL is not configured.");
      }

      const response = await fetch(`${API_URL.replace(/\/$/, "")}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input }),
      });
      const payload = await response.json().catch(() => null);

      if (!response.ok) {
        const detail = payload?.detail;
        throw new Error(
          typeof detail === "string"
            ? detail
            : "Axiom could not complete this analysis.",
        );
      }

      setResult(payload as AnalyzeResponse);
    } catch (requestError) {
      setError(getErrorMessage(requestError));
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#f2efe7] text-[#17201c]">
      <div className="border-b border-[#17201c]/15 bg-[#17201c] text-[#f7f4ec]">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 sm:px-8 lg:px-12">
          <a
            href="#top"
            className="text-sm font-bold tracking-[0.28em] uppercase"
          >
            Axiom
          </a>
          <p className="hidden text-xs tracking-[0.16em] text-[#cbd5ce] uppercase sm:block">
            Evidence-led claim analysis
          </p>
          <span className="flex items-center gap-2 text-xs font-medium text-[#c7f36b]">
            <span className="h-2 w-2 rounded-full bg-[#c7f36b]" />
            Live analysis
          </span>
        </div>
      </div>

      <div id="top" className="mx-auto max-w-7xl px-5 sm:px-8 lg:px-12">
        <header className="grid gap-8 border-x border-[#17201c]/15 px-5 pt-12 pb-10 sm:px-10 sm:pt-16 sm:pb-12 lg:grid-cols-[1fr_0.8fr] lg:items-end lg:px-14 lg:pt-18 lg:pb-14">
          <div>
            <p className="mb-5 font-mono text-xs font-semibold tracking-[0.22em] text-[#d54b2a] uppercase">
              Claim intelligence / 01
            </p>
            <h1 className="max-w-3xl text-4xl leading-[1.02] font-semibold tracking-[-0.045em] sm:text-5xl lg:text-6xl">
              See what the evidence says.
            </h1>
          </div>
          <div className="flex items-end">
            <p className="max-w-lg text-base leading-7 text-[#526059] sm:text-lg sm:leading-8">
              Paste a statement and Axiom will break it into claims, retrieve
              reliable sources, weigh contradictions, and show its reasoning.
            </p>
          </div>
        </header>

        <section className="border border-[#17201c]/15 bg-[#fcfaf4] p-5 shadow-[8px_8px_0_#17201c] sm:p-8 lg:p-10">
          <form onSubmit={handleSubmit}>
            <div className="mb-5 flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
              <div>
                <label
                  htmlFor="claim-text"
                  className="text-sm font-semibold tracking-wide"
                >
                  Text to analyze
                </label>
                <p className="mt-1 text-sm text-[#647169]">
                  One claim or a short paragraph works best.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <span className="mr-1 text-xs font-medium text-[#6d776f] uppercase">
                  Try an example
                </span>
                {presets.map((preset) => (
                  <button
                    key={preset.label}
                    type="button"
                    onClick={() => {
                      setText(preset.text);
                      setError(null);
                    }}
                    className="rounded-full border border-[#17201c]/25 px-3 py-1.5 text-xs font-semibold transition hover:border-[#17201c] hover:bg-[#17201c] hover:text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#d54b2a]"
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>

            <textarea
              id="claim-text"
              value={text}
              maxLength={20000}
              onChange={(event) => setText(event.target.value)}
              placeholder="Example: Earthquake magnitude is measured on a logarithmic scale."
              className="min-h-52 w-full resize-y border border-[#17201c]/30 bg-white p-5 text-lg leading-8 outline-none transition placeholder:text-[#9ba39e] focus:border-[#d54b2a] focus:ring-2 focus:ring-[#d54b2a]/15 sm:min-h-60 sm:p-6 sm:text-xl"
            />

            <div className="mt-4 flex flex-col-reverse gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="min-h-6" aria-live="polite">
                {error ? (
                  <p className="flex items-center gap-2 text-sm font-medium text-[#a52f1b]">
                    <span aria-hidden="true">●</span>
                    {error}
                  </p>
                ) : (
                  <p className="text-xs text-[#77817b]">
                    {text.length.toLocaleString()} / 20,000 characters
                  </p>
                )}
              </div>
              <button
                type="submit"
                disabled={isLoading}
                className="inline-flex min-h-12 items-center justify-center gap-3 bg-[#d54b2a] px-6 py-3 text-sm font-bold tracking-wide text-white transition hover:bg-[#b83b20] disabled:cursor-wait disabled:bg-[#9b776e] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#17201c]"
              >
                {isLoading ? "Analyzing claims" : "Run evidence check"}
                <span aria-hidden="true">→</span>
              </button>
            </div>
          </form>
        </section>

        {isLoading && (
          <section
            className="my-10 border border-[#17201c]/15 bg-[#e7e2d6] p-6 sm:p-8"
            aria-live="polite"
            aria-busy="true"
          >
            <div className="flex items-start gap-4">
              <span className="mt-1 block h-3 w-3 animate-pulse rounded-full bg-[#d54b2a]" />
              <div>
                <h2 className="font-semibold">Following the evidence trail</h2>
                <p className="mt-2 max-w-2xl text-sm leading-6 text-[#5f6a63]">
                  Extracting claims, finding sources, and weighing
                  contradictions. The first request may take up to a minute
                  while the backend wakes up.
                </p>
              </div>
            </div>
          </section>
        )}

        {result && (
          <section className="py-14 sm:py-20" aria-labelledby="results-heading">
            <div className="mb-8 flex flex-col justify-between gap-4 border-b border-[#17201c]/20 pb-5 sm:flex-row sm:items-end">
              <div>
                <p className="font-mono text-xs font-semibold tracking-[0.2em] text-[#d54b2a] uppercase">
                  Analysis complete
                </p>
                <h2
                  id="results-heading"
                  className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl"
                >
                  {result.claims.length} claim
                  {result.claims.length === 1 ? "" : "s"} examined
                </h2>
              </div>
              <p className="font-mono text-xs text-[#6f7973]">
                Report {result.submission_id.slice(0, 8)}
              </p>
            </div>

            <div className="grid gap-6">
              {result.claims.map((claim, index) => (
                <ClaimCard
                  key={claim.id}
                  index={index + 1}
                  claim={claim}
                  verdict={result.verdicts.find(
                    (item) => item.claim_id === claim.id,
                  )}
                  evidence={result.evidence_by_claim[claim.id] ?? []}
                />
              ))}
            </div>
          </section>
        )}

        <footer className="mt-10 flex flex-col gap-5 border-x border-t border-[#17201c]/15 px-5 py-8 text-sm text-[#68736c] sm:flex-row sm:items-center sm:justify-between sm:px-10">
          <p>Axiom shows its sources. You make the final call.</p>
          <div className="flex gap-5 font-mono text-xs uppercase">
            <span>Retrieve</span>
            <span>Verify</span>
            <span>Explain</span>
          </div>
        </footer>
      </div>
    </main>
  );
}
