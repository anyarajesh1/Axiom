export type Claim = {
  id: string;
  text: string;
  source_span: string | null;
};

export type Evidence = {
  id: string;
  text: string;
  source_name: string;
  source_url: string;
  category: string;
  score: number;
  reranker_score: number | null;
  combined_score: number | null;
};

export type VerdictLabel =
  | "supported"
  | "contradicted"
  | "insufficient_evidence";

export type Verdict = {
  claim_id: string;
  label: VerdictLabel;
  confidence: number;
  evidence_ids: string[];
  explanation: string;
};

export type AnalyzeResponse = {
  submission_id: string;
  claims: Claim[];
  evidence_by_claim: Record<string, Evidence[]>;
  verdicts: Verdict[];
};
