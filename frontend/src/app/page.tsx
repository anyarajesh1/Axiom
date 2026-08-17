type DependencyStatus = "green" | "red";

type HealthResponse = {
  status: "ok" | "degraded";
  dependencies: {
    groq: DependencyStatus;
    qdrant: DependencyStatus;
    neon: DependencyStatus;
  };
};

async function getHealth(): Promise<HealthResponse | null> {
  try {
    const response = await fetch(
      process.env.AXIOM_API_URL ?? "http://127.0.0.1:8000/health/deps",
      { cache: "no-store" },
    );

    if (!response.ok) {
      return null;
    }

    return response.json();
  } catch {
    return null;
  }
}

export default async function Home() {
  const health = await getHealth();
  const dependencies = health?.dependencies ?? {
    groq: "red",
    qdrant: "red",
    neon: "red",
  };

  const services = [
    { id: "groq", name: "Groq", purpose: "Language model inference" },
    { id: "qdrant", name: "Qdrant", purpose: "Vector search" },
    { id: "neon", name: "Neon", purpose: "PostgreSQL database" },
  ] as const;

  return (
    <main className="min-h-screen bg-slate-950 px-6 py-16 text-slate-100">
      <div className="mx-auto max-w-5xl">
        <header className="mb-14">
          <p className="mb-4 text-sm font-semibold tracking-[0.3em] text-cyan-400">
            AXIOM
          </p>
          <h1 className="max-w-3xl text-4xl font-semibold tracking-tight sm:text-6xl">
            Trace claims back to what is true.
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-400">
            Axiom evaluates general knowledge, technology, science, and
            AI-generated claims against reliable sources.
          </p>
        </header>

        <section className="rounded-3xl border border-slate-800 bg-slate-900/60 p-6 sm:p-8">
          <div className="mb-8 flex items-center justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold">System status</h2>
              <p className="mt-1 text-sm text-slate-400">
                Live dependency checks from the Axiom API
              </p>
            </div>
            <span
              className={`rounded-full px-3 py-1 text-sm font-medium ${
                health?.status === "ok"
                  ? "bg-emerald-500/15 text-emerald-300"
                  : "bg-red-500/15 text-red-300"
              }`}
            >
              {health?.status === "ok" ? "All systems ready" : "Needs attention"}
            </span>
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            {services.map((service) => {
              const isGreen = dependencies[service.id] === "green";

              return (
                <article
                  key={service.id}
                  className="rounded-2xl border border-slate-800 bg-slate-950/70 p-5"
                >
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold">{service.name}</h3>
                    <span
                      className={`h-3 w-3 rounded-full ${
                        isGreen ? "bg-emerald-400" : "bg-red-400"
                      }`}
                    />
                  </div>
                  <p className="mt-2 text-sm text-slate-400">
                    {service.purpose}
                  </p>
                  <p
                    className={`mt-5 text-sm font-medium ${
                      isGreen ? "text-emerald-300" : "text-red-300"
                    }`}
                  >
                    {isGreen ? "Connected" : "Unavailable"}
                  </p>
                </article>
              );
            })}
          </div>
        </section>
      </div>
    </main>
  );
}
