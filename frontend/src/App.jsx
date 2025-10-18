import React, { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE?.trim();

function cls(...a) { return a.filter(Boolean).join(" "); }
function isApiConfigured() { return !!API_BASE && !API_BASE.includes("<"); }
function prettyPrice(x) {
  if (x === undefined || x === null || x === "" || isNaN(Number(x))) return "—";
  const n = Number(x);
  return n < 1000 ? `$${n.toFixed(2)}` : `$${n.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
}

/* ————— Theme Toggle ————— */
function useTheme() {
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem("theme");
    if (saved) return saved === "dark";
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  });
  useEffect(() => {
    const root = document.documentElement;
    if (dark) {
      root.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      root.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [dark]);
  return { dark, setDark };
}

function ThemeToggle({ dark, onToggle }) {
  return (
    <button
      type="button"
      onClick={() => onToggle(!dark)}
      className="h-9 w-9 rounded-xl border border-neutral-200/70 dark:border-neutral-800 grid place-items-center
                 bg-white/70 dark:bg-neutral-900/70 hover:bg-white dark:hover:bg-neutral-800 transition"
      title={dark ? "Switch to light" : "Switch to dark"}
    >
      {dark ? (
        /* Sun */
        <svg width="18" height="18" viewBox="0 0 24 24" className="text-amber-300"><path fill="currentColor" d="M6.76 4.84l-1.8-1.79L3.17 4.84l1.79 1.79l1.8-1.79m10.48 0l1.79-1.79l1.79 1.79l-1.79 1.79l-1.79-1.79M12 4V1h-0v3h0m0 19v-3h-0v3h0M4 13H1v-0h3v0m22 0h-3v-0h3v0M6.76 19.16l-1.8 1.79l-1.79-1.79l1.79-1.79l1.8 1.79m10.48 0l1.79 1.79l1.79-1.79l-1.79-1.79l-1.79 1.79M12 8a4 4 0 100 8a4 4 0 000-8z" /></svg>
      ) : (
        /* Moon */
        <svg width="18" height="18" viewBox="0 0 24 24" className="text-neutral-700"><path fill="currentColor" d="M12.75 2a9 9 0 1 0 9.25 10.58A8 8 0 0 1 12.75 2z" /></svg>
      )}
    </button>
  );
}

/* ————— Brand Chip ————— */
function Chip({ label, active, onClick }) {
  return (
    <button
      onClick={onClick}
      title={label}
      className={cls(
        "px-3.5 py-1.5 rounded-full text-xs font-medium border transition",
        active
          ? "bg-neutral-900 text-white border-neutral-900 dark:bg-white dark:text-neutral-900 dark:border-white"
          : "bg-white border-neutral-200 text-neutral-700 hover:bg-neutral-50 dark:bg-neutral-900 dark:border-neutral-800 dark:text-neutral-300 dark:hover:bg-neutral-800"
      )}
    >
      {label}
    </button>
  );
}

export default function App() {
  const { dark, setDark } = useTheme();

  const [q, setQ] = useState("sofa");
  const [topK, setTopK] = useState(8);
  const [brand, setBrand] = useState("all");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [hits, setHits] = useState([]);
  const [text, setText] = useState("");
  const [brands, setBrands] = useState([]);
  const [analytics, setAnalytics] = useState({ products_per_brand: {}, avg_price_per_brand: {} });

  const showEnvWarning = !isApiConfigured();

  /* compute filtered hits by brand chip */
  const filtered = useMemo(() => {
    if (brand === "all") return hits;
    return hits.filter((h) => (h.brand || "").toLowerCase() === brand.toLowerCase());
  }, [hits, brand]);

  useEffect(() => {
    async function loadAnalytics() {
      try {
        if (!isApiConfigured()) return;
        const r = await fetch(`${API_BASE}/analytics`);
        if (!r.ok) throw new Error(`Analytics ${r.status}`);
        const data = await r.json();
        setAnalytics(data || {});
        const brandList = Object.keys(data?.products_per_brand || {});
        setBrands(brandList);
      } catch (e) {
        console.error(e);
      }
    }
    loadAnalytics();
  }, []);

  async function runSearch(e) {
    e?.preventDefault?.();
    setErr("");
    setLoading(true);
    setHits([]);
    setText("");

    try {
      if (!isApiConfigured()) throw new Error("API base URL is not set. Add VITE_API_BASE in frontend/.env");
      const r = await fetch(`${API_BASE}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, top_k: Number(topK) }),
      });
      if (!r.ok) throw new Error(`API ${r.status}`);
      const data = await r.json();
      setHits(data?.recommendations || []);
      setText(data?.generated_text || "");
      if (!brands.length) {
        const uniq = Array.from(new Set((data?.recommendations || [])
          .map((x) => (x.brand || "").trim())
          .filter(Boolean)));
        setBrands(uniq);
      }
    } catch (e) {
      console.error(e);
      setErr(e.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-neutral-50 dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100">
      {/* Header */}
      <header className="sticky top-0 z-40 w-full border-b border-neutral-200/70 dark:border-neutral-800 bg-white/70 dark:bg-neutral-900/70 backdrop-blur">
        <div className="mx-auto max-w-7xl px-4 sm:px-6">
          <div className="h-16 flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="h-9 w-9 rounded-2xl bg-gradient-to-br from-teal-500 to-blue-600 grid place-items-center text-white font-semibold">F</div>
              <div>
                <div className="font-semibold">FurniFind</div>
                <div className="text-xs text-neutral-500 dark:text-neutral-400 -mt-0.5">Smarter furniture picks, powered by Pinecone + HF</div>
              </div>
            </div>
            <ThemeToggle dark={dark} onToggle={setDark} />
          </div>
        </div>
      </header>

      {/* Env banner */}
      {showEnvWarning && (
        <div className="bg-amber-50 dark:bg-amber-900/20 border-y border-amber-200 dark:border-amber-800 text-amber-900 dark:text-amber-200">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 py-3 text-sm">
            <strong>Setup needed:</strong> Set <code className="bg-amber-100 dark:bg-amber-800/50 px-1 rounded">VITE_API_BASE</code> in <code className="bg-amber-100 dark:bg-amber-800/50 px-1 rounded">frontend/.env</code> to your Render URL.
          </div>
        </div>
      )}

      {/* Hero Gradient */}
      <section className="relative">
        <div className="absolute inset-0 -z-10 bg-gradient-to-br from-sky-500/15 via-fuchsia-500/10 to-emerald-500/15 dark:from-sky-400/10 dark:via-fuchsia-400/10 dark:to-emerald-400/10" />
        <div className="mx-auto max-w-7xl px-4 sm:px-6 py-10">
          <div className="max-w-3xl">
            <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">
              Find furniture that just fits.
            </h1>
            <p className="mt-2 text-neutral-600 dark:text-neutral-300">
              Ask for sofas, chairs, ottomans, benches… we’ll fetch relevant picks and summarize them for you.
            </p>
          </div>

          {/* Search + Controls inside hero */}
          <form onSubmit={runSearch} className="mt-6 flex flex-wrap items-center gap-2">
            <div className="relative flex-1">
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Search sofas, chairs, ottomans, benches…"
                className="w-full h-12 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900
                           px-4 pr-10 outline-none focus:ring-4 ring-blue-100 dark:ring-blue-900/40"
              />
              <div className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-400 dark:text-neutral-600">⌘K</div>
            </div>

            <select
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
              className="h-12 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 px-3 text-sm"
              title="How many results"
            >
              {[5, 8, 10, 12].map((n) => <option key={n} value={n}>{n}</option>)}
            </select>

            <button
              type="submit"
              className={cls(
                "h-12 rounded-xl px-5 text-white text-sm font-medium transition",
                loading ? "bg-neutral-400 cursor-not-allowed" : "bg-neutral-900 hover:bg-neutral-800 dark:bg-white dark:text-neutral-900 dark:hover:bg-neutral-200"
              )}
              disabled={loading}
            >
              {loading ? "Searching…" : "Search"}
            </button>
          </form>

          {/* Brand chips */}
          <div className="mt-4 overflow-x-auto no-scrollbar">
            <div className="flex items-center gap-2 min-w-max">
              <Chip label="All" active={brand === "all"} onClick={() => setBrand("all")} />
              {brands.map((b) => (
                <Chip key={b} label={b} active={brand.toLowerCase() === b.toLowerCase()} onClick={() => setBrand(b)} />
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Main content */}
      <main className="mx-auto max-w-7xl w-full px-4 sm:px-6 py-6 grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1">
        {/* Left: Results */}
        <section className="lg:col-span-2">
          <h2 className="text-sm font-semibold text-neutral-700 dark:text-neutral-300 mb-3">Top Picks</h2>

          {/* Error */}
          {err && (
            <div className="rounded-xl border border-rose-200 dark:border-rose-900 bg-rose-50 dark:bg-rose-900/20 text-rose-800 dark:text-rose-200 p-3 text-sm mb-3">
              {err}
            </div>
          )}

          {/* Empty state */}
          {!loading && !err && filtered.length === 0 && (
            <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-10 text-center text-neutral-500 dark:text-neutral-400">
              No results yet. Try a broader query.
            </div>
          )}

          {/* Skeletons */}
          {loading && (
            <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-4">
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-3">
                  <div className="aspect-[4/3] rounded-lg bg-neutral-100 dark:bg-neutral-800 animate-pulse mb-3"></div>
                  <div className="h-4 w-4/5 bg-neutral-100 dark:bg-neutral-800 rounded mb-2 animate-pulse"></div>
                  <div className="h-4 w-2/3 bg-neutral-100 dark:bg-neutral-800 rounded mb-4 animate-pulse"></div>
                  <div className="h-3 w-1/3 bg-neutral-100 dark:bg-neutral-800 rounded mb-1 animate-pulse"></div>
                  <div className="h-3 w-1/4 bg-neutral-100 dark:bg-neutral-800 rounded animate-pulse"></div>
                </div>
              ))}
            </div>
          )}

          {/* Cards */}
          {!loading && filtered.length > 0 && (
            <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-4">
              {filtered.map((it) => (
                <article key={it.id} className="group rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-3 shadow-sm hover:shadow-md transition">
                  <div className="relative">
                    <img
                      src={it.image_url}
                      alt={it.title}
                      loading="lazy"
                      className="aspect-[4/3] w-full rounded-lg object-cover bg-neutral-100 dark:bg-neutral-800"
                      onError={(e) => (e.currentTarget.style.opacity = 0.2)}
                    />
                    {it.score !== undefined && (
                      <div className="absolute bottom-2 right-2 text-[10px] px-2 py-1 rounded-full bg-black/70 text-white">
                        score {Number(it.score).toFixed(3)}
                      </div>
                    )}
                  </div>
                  <h3 className="mt-3 text-sm font-medium leading-5 line-clamp-2">{it.title}</h3>
                  <div className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">{it.brand || "—"}</div>
                  <div className="mt-2 text-sm font-semibold">{prettyPrice(it.price)}</div>
                  <div className="mt-3">
                    <a
                      href={it.image_url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-2 text-xs rounded-lg border border-neutral-200 dark:border-neutral-700 px-3 py-1.5 hover:bg-neutral-50 dark:hover:bg-neutral-800"
                    >
                      View image
                      <svg width="12" height="12" viewBox="0 0 24 24" className="opacity-70"><path d="M14 3h7v7h-2V6.41l-9.29 9.3-1.42-1.42 9.3-9.29H14V3zM5 5h5V3H3v7h2V5zM5 19v-5H3v7h7v-2H5z" fill="currentColor" /></svg>
                    </a>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        {/* Right: Assistant + Analytics */}
        <aside className="lg:col-span-1 space-y-6">
          <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-4">
            <h3 className="text-sm font-semibold text-neutral-700 dark:text-neutral-300 mb-2">Assistant</h3>
            {loading ? (
              <div className="space-y-2">
                <div className="h-4 rounded bg-neutral-100 dark:bg-neutral-800 animate-pulse"></div>
                <div className="h-4 rounded bg-neutral-100 dark:bg-neutral-800 animate-pulse"></div>
                <div className="h-4 rounded bg-neutral-100 dark:bg-neutral-800 animate-pulse"></div>
              </div>
            ) : (
              <p className="text-sm leading-6 text-neutral-700 dark:text-neutral-300 whitespace-pre-wrap">
                {text || "Ask for sofas, chairs, ottomans, benches…"}
              </p>
            )}
          </div>

          <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-4">
            <h3 className="text-sm font-semibold text-neutral-700 dark:text-neutral-300 mb-3">Analytics</h3>

            {/* Products per brand */}
            <div className="text-xs font-medium text-neutral-500 dark:text-neutral-400 mb-1">Top brands by count</div>
            <div className="space-y-2 mb-4">
              {Object.entries(analytics.products_per_brand || {}).slice(0, 6).map(([b, v], i, arr) => {
                const max = Math.max(...arr.map(([, val]) => val || 0), 1);
                const pct = Math.round((v / max) * 100);
                return (
                  <div key={b}>
                    <div className="flex justify-between text-xs">
                      <span className="truncate pr-2">{b}</span>
                      <span>{v}</span>
                    </div>
                    <div className="h-2 rounded bg-neutral-100 dark:bg-neutral-800">
                      <div className="h-2 rounded bg-neutral-800 dark:bg-neutral-200" style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Avg price */}
            <div className="text-xs font-medium text-neutral-500 dark:text-neutral-400 mb-1">Avg price by brand</div>
            <div className="space-y-2">
              {Object.entries(analytics.avg_price_per_brand || {}).slice(0, 6).map(([b, v], i, arr) => {
                const max = Math.max(...arr.map(([, val]) => val || 0), 1);
                const pct = Math.round((v / max) * 100);
                return (
                  <div key={b}>
                    <div className="flex justify-between text-xs">
                      <span className="truncate pr-2">{b}</span>
                      <span>{prettyPrice(v)}</span>
                    </div>
                    <div className="h-2 rounded bg-neutral-100 dark:bg-neutral-800">
                      <div className="h-2 rounded bg-neutral-800 dark:bg-neutral-200" style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </aside>
      </main>

      {/* Footer */}
      <footer className="border-t border-neutral-200/70 dark:border-neutral-800 bg-white dark:bg-neutral-900">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 py-6 text-center text-sm text-neutral-500 dark:text-neutral-400">
          Built with FastAPI • Pinecone • HF Inference
          <div className="mt-2 text-neutral-600 dark:text-neutral-300">
            Made with <span aria-label="love" title="love">❤️</span> by <strong>Devansh</strong> for <strong>Ikarus 3D</strong>
          </div>
        </div>
      </footer>
    </div>
  );
}
