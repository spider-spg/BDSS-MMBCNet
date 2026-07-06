import { useState, useEffect, useMemo, useRef } from 'react';
import type { ReactNode, ChangeEvent, DragEvent } from 'react';

const logoUrl =
  'https://lh3.googleusercontent.com/aida-public/AB6AXuDOZrcFB16kxK0xFc2gjIrv9NjV6bK0QnKtKG4Z5gWXkRv6bUOK-vm07VWqp2Fxy9K-lp1a-riq77Bh932mbjeVBzbX9yMdE7TyzdaiVWCo1glLtVIkSIJGtap67k_3hnIoQhXNX5v204IgLPHbXUN1dHObtXYQEUoFoh2BEoXmhxsrEa_AaBUXYytAtWpWjlbqFueuxJEM9W9AlSIrc_BOlO7YUsdoS6xe-aP713aR0niSU9UlC7d5V8gEoJhFggUm_5c-5Gq-_Qs';

const apiBase = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000').replace(/\/$/, '');

type Modality = 'mammo' | 'sono';

type DicomMetadata = {
  patient_name: string;
  patient_id: string;
  patient_dob: string;
  patient_sex: string;
  study_date: string;
  study_description: string;
  modality: string;
  accession_number: string;
  institution: string;
  referring_physician: string;
};

type PredictionResult = {
  modality: string;
  filename: string;
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  flagged: boolean;
  flag_reasons: string[];
  gradcam_image_b64: string;
  dicom_metadata: DicomMetadata | null;
};

function IconButton({ children, ariaLabel, onClick }: { children: ReactNode; ariaLabel: string; onClick?: () => void }) {
  return (
    <button
      type="button"
      aria-label={ariaLabel}
      onClick={onClick}
      className="flex h-9 w-9 items-center justify-center rounded border border-[#c4c5d7] bg-white text-[#434655] shadow-sm transition hover:bg-[#ededf9]"
    >
      {children}
    </button>
  );
}

function Panel({ title, icon, children, className = '' }: { title: string; icon?: string; children: ReactNode; className?: string }) {
  return (
    <section className={`overflow-hidden rounded border border-[#c4c5d7] bg-white shadow-sm ${className}`.trim()}>
      <header className="flex items-center gap-2 border-b border-[#c4c5d7] bg-[#f2f0fb] px-4 py-3">
        {icon ? <span className="material-symbols-outlined text-[18px] text-[#434655]">{icon}</span> : null}
        <h3 className="font-semibold text-[16px] text-[#2c2f3a]">{title}</h3>
      </header>
      {children}
    </section>
  );
}

function Dashboard({ onSignOut }: { onSignOut: () => void }) {
  const [profileOpen, setProfileOpen] = useState(false);
  const [previewExpanded, setPreviewExpanded] = useState(false);
  const [modality, setModality] = useState<Modality>('mammo');
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [health, setHealth] = useState<'unknown' | 'ok' | 'down'>('unknown');
  const [isDicom, setIsDicom] = useState(false);

  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  useEffect(() => {
    let active = true;
    fetch(`${apiBase}/health`)
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error('unhealthy'))))
      .then(() => {
        if (active) {
          setHealth('ok');
        }
      })
      .catch(() => {
        if (active) {
          setHealth('down');
        }
      });

    return () => {
      active = false;
    };
  }, []);

  const probabilityRows = useMemo(() => {
    if (!result) {
      return [];
    }

    return Object.entries(result.probabilities)
      .map(([label, value]) => ({ label, value }))
      .sort((left, right) => right.value - left.value);
  }, [result]);

  const confidencePercent = result ? (result.confidence * 100).toFixed(1) : '0.0';

  const setSelectedFile = (nextFile: File | null) => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setFile(nextFile);
    setResult(null);
    setError('');
    setIsDicom(false);

    if (nextFile) {
      // For DICOM files, we can't render a native preview — show a placeholder
      const dcm = nextFile.name.toLowerCase().endsWith('.dcm');
      setIsDicom(dcm);
      if (!dcm) {
        setPreviewUrl(URL.createObjectURL(nextFile));
      } else {
        setPreviewUrl(''); // Preview generated after analysis
      }
      setPreviewExpanded(false);
    } else {
      setPreviewUrl('');
    }
  };

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    setSelectedFile(event.target.files?.[0] ?? null);
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files?.[0] ?? null;
    if (droppedFile) {
      setSelectedFile(droppedFile);
    }
  };

  const runAnalysis = async () => {
    if (!file) {
      setError('Choose an image first.');
      return;
    }

    setBusy(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${apiBase}/predict/${modality}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(payload?.detail ?? 'Prediction request failed.');
      }

      const payload = (await response.json()) as PredictionResult;
      setResult(payload);
    } catch (analysisError) {
      setError(analysisError instanceof Error ? analysisError.message : 'Unexpected error during analysis.');
    } finally {
      setBusy(false);
    }
  };

  const clearAll = () => {
    setSelectedFile(null);
    setResult(null);
    setError('');
    setBusy(false);
    setIsDicom(false);
  };

  // Format DICOM date string YYYYMMDD → DD/MM/YYYY
  const fmtDate = (d: string) => {
    if (!d || d.length < 8) return d;
    return `${d.slice(6, 8)}/${d.slice(4, 6)}/${d.slice(0, 4)}`;
  };

  const handlePrint = () => window.print();

  const handleExportPdf = () => {
    // Trigger browser print dialog — user can "Save as PDF"
    window.print();
  };

  const gradcamDataUrl = result ? `data:image/png;base64,${result.gradcam_image_b64}` : '';

  return (
    <div className="flex min-h-screen flex-col bg-[#faf8ff] text-[#1a1b23]">
      <header className="z-10 flex h-16 w-full shrink-0 items-center justify-between border-b border-[#c4c5d7] bg-white px-5 shadow-sm">
        <div className="flex items-center gap-4">
          <div className="flex h-10 items-center">
            <img src={logoUrl} alt="BIDSS Logo" className="h-full w-auto object-contain" />
          </div>
          <div className="h-6 w-px bg-[#c4c5d7]" />
          <nav className="flex h-full gap-3">
            <a href="#" className="flex h-full items-center border-b-2 border-[#0037b0] px-2 font-semibold text-[#0037b0]">
              Diagnostic Support
            </a>
          </nav>
        </div>

        <div className="relative flex items-center gap-4">
          <div className="rounded-full border border-[#c4c5d7] bg-[#f9fafb] px-3 py-1 text-[12px] font-semibold text-[#575e70]">
            Backend: {health === 'ok' ? 'online' : health === 'down' ? 'offline' : 'checking'}
          </div>

          <button
            type="button"
            aria-label="Open profile menu"
            aria-expanded={profileOpen}
            onClick={() => setProfileOpen((open) => !open)}
            className="flex items-center gap-3 rounded px-1 py-1 text-left transition hover:bg-[#f3f2fe]"
          >
            <div className="flex flex-col items-end leading-tight">
              <span className="font-semibold text-[14px] text-[#2c2f3a]">Welcome, Dr. Priya Sharma</span>
              <span className="mt-1 rounded bg-[#e8e7f3] px-2 py-[2px] text-[12px] font-semibold text-[#5c6274]">Radiologist</span>
            </div>
            <div className="flex h-8 w-8 items-center justify-center overflow-hidden rounded-full border border-[#c4c5d7] bg-[#e8e7f3] text-[#434655]">
              <span className="material-symbols-outlined text-[18px]">person</span>
            </div>
          </button>

          {profileOpen ? (
            <div className="absolute right-0 top-[calc(100%+10px)] z-20 w-72 overflow-hidden rounded border border-[#c4c5d7] bg-white shadow-[0_14px_32px_rgba(17,24,39,0.14)]">
              <div className="border-b border-[#c4c5d7] bg-[#f9fafb] px-4 py-3">
                <div className="text-[14px] font-semibold text-[#1a1b23]">Dr. Priya Sharma</div>
                <div className="mt-1 text-[12px] text-[#575e70]">priya.sharma@clinic.com</div>
              </div>
              <div className="space-y-1 px-4 py-3 text-[13px] text-[#434655]">
                <div className="flex justify-between gap-4">
                  <span className="text-[#747686]">Role</span>
                  <span className="font-semibold text-[#1a1b23]">Radiologist</span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-[#747686]">Department</span>
                  <span className="font-semibold text-[#1a1b23]">Breast Imaging</span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-[#747686]">Last login</span>
                  <span className="font-semibold text-[#1a1b23]">Today, 09:24 AM</span>
                </div>
              </div>
              <div className="border-t border-[#c4c5d7] p-2">
                <button
                  type="button"
                  aria-label="Log out"
                  onClick={onSignOut}
                  className="flex w-full items-center justify-center rounded px-3 py-2 text-[13px] font-semibold text-[#ba1a1a] transition hover:bg-[#ffdad6]/40"
                >
                  Logout
                </button>
              </div>
            </div>
          ) : null}
        </div>
      </header>

      <div className="flex min-h-0 flex-1 overflow-hidden">
        <aside className="flex w-[280px] shrink-0 flex-col overflow-y-auto border-r border-[#c4c5d7] bg-white">
          <div className="flex flex-col gap-6 p-4">
            <section>
              <h2 className="mb-2 text-[12px] font-semibold uppercase tracking-[0.14em] text-[#575e70]">Action</h2>
              <button
                type="button"
                aria-label="Start New Analysis"
                onClick={clearAll}
                className="flex h-11 w-full items-center justify-center gap-2 rounded border border-[#1d4ed8] bg-white px-4 text-[14px] font-semibold text-[#1d4ed8] transition hover:bg-[#f3f2fe]"
              >
                <span className="text-[18px] leading-none">+</span>
                Start New Analysis
              </button>
            </section>

            <div className="h-px bg-[#c4c5d7]" />

            <section>
              <h2 className="mb-3 text-[12px] font-semibold uppercase tracking-[0.14em] text-[#575e70]">Modality</h2>
              <div className="space-y-3">
                <label className="flex cursor-pointer items-center gap-2">
                  <input
                    type="radio"
                    name="modality"
                    checked={modality === 'mammo'}
                    onChange={() => setModality('mammo')}
                    className="h-4 w-4 accent-[#0037b0]"
                  />
                  <span className="text-[14px]">Mammography</span>
                </label>
                <label className="flex cursor-pointer items-center gap-2">
                  <input
                    type="radio"
                    name="modality"
                    checked={modality === 'sono'}
                    onChange={() => setModality('sono')}
                    className="h-4 w-4 accent-[#0037b0]"
                  />
                  <span className="text-[14px]">Ultrasound</span>
                </label>
              </div>
            </section>

            <div className="h-px bg-[#c4c5d7]" />

            <section>
              <h2 className="mb-3 flex items-center justify-between text-[12px] font-semibold uppercase tracking-[0.14em] text-[#575e70]">
                <span>Uploaded Image</span>
                <button type="button" onClick={clearAll} className="text-[10px] font-semibold lowercase text-[#0037b0] hover:underline">
                  clear
                </button>
              </h2>

              <input ref={inputRef} type="file" accept="image/*,.dcm" className="hidden" onChange={onFileChange} />

              <div
                role="button"
                tabIndex={0}
                onClick={() => inputRef.current?.click()}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    inputRef.current?.click();
                  }
                }}
                onDragOver={(event) => event.preventDefault()}
                onDrop={onDrop}
                className="group flex aspect-square w-full cursor-pointer items-center justify-center overflow-hidden rounded border-2 border-dashed border-[#c4c5d7] bg-[#f3f2fe] transition hover:border-[#1d4ed8] hover:bg-[#ededf9]"
              >
                {previewUrl && !isDicom ? (
                  <img src={previewUrl} alt="Selected upload preview" className="h-full w-full object-contain" />
                ) : isDicom ? (
                  <div className="flex flex-col items-center gap-2 px-4 text-center">
                    <span className="material-symbols-outlined text-[36px] text-[#0037b0]">description</span>
                    <div className="text-[13px] font-semibold text-[#1a1b23]">{file?.name}</div>
                    <div className="text-[11px] text-[#575e70]">DICOM file ready · Run analysis to preview</div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-3 px-4 text-center text-[#575e70] transition group-hover:text-[#1a1b23]">
                    <div className="flex h-12 w-12 items-center justify-center rounded-full border border-[#c4c5d7] bg-white shadow-sm">
                      <span className="material-symbols-outlined text-[22px] text-[#0037b0]">upload</span>
                    </div>
                    <div>
                      <div className="text-[14px] font-semibold text-[#1a1b23]">Upload Image</div>
                      <div className="mt-1 text-[12px] text-[#747686]">JPG · PNG · DICOM (.dcm)</div>
                      <div className="mt-0.5 text-[11px] text-[#a0a3ae]">Drag and drop or click to browse</div>
                    </div>
                  </div>
                )}
              </div>

              {file ? <div className="mt-2 text-[12px] text-[#575e70] truncate">{file.name}</div> : null}
            </section>

            <section className="mt-auto pt-2">
              <button
                type="button"
                onClick={runAnalysis}
                disabled={!file || busy}
                className="flex h-11 w-full items-center justify-center gap-2 rounded bg-[#0037b0] px-4 text-[14px] font-semibold text-white shadow-sm transition hover:bg-[#1d4ed8] disabled:cursor-not-allowed disabled:opacity-60"
              >
                {busy ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" aria-hidden="true" /> : (
                  <span className="material-symbols-outlined text-[16px]">analytics</span>
                )}
                {busy ? 'Running Analysis' : 'Run Analysis'}
              </button>
            </section>
          </div>
        </aside>

        <main className="relative flex min-w-0 flex-1 flex-col overflow-y-auto p-6">
          <div className="absolute left-0 right-0 top-0 h-1 overflow-hidden bg-[#e2e1ed]">
            <div className="h-full w-full origin-left scale-x-100 bg-[#0037b0]" />
          </div>

          <div className="flex items-end justify-between border-b border-[#c4c5d7] pb-2">
            <div>
              <h1 className="text-[18px] font-semibold leading-7 text-[#1a1b23]">Analysis Results</h1>
            </div>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={handleExportPdf}
                disabled={!result}
                className="flex items-center gap-1 rounded border border-[#c4c5d7] bg-white px-3 py-1.5 text-[14px] font-semibold text-[#575e70] shadow-sm hover:bg-[#f3f2fe] disabled:opacity-40 disabled:cursor-not-allowed print:hidden"
              >
                <span className="material-symbols-outlined text-[16px]">download</span>
                Export PDF
              </button>
              <button
                type="button"
                onClick={handlePrint}
                disabled={!result}
                className="flex items-center gap-1 rounded border border-[#c4c5d7] bg-white px-3 py-1.5 text-[14px] font-semibold text-[#575e70] shadow-sm hover:bg-[#f3f2fe] disabled:opacity-40 disabled:cursor-not-allowed print:hidden"
              >
                <span className="material-symbols-outlined text-[16px]">print</span>
                Print
              </button>
            </div>
          </div>

          {/* Patient Info Panel — shown when DICOM metadata is available */}
          {result?.dicom_metadata && (() => {
            const m = result.dicom_metadata;
            const rows = [
              { label: 'Patient Name',       value: m.patient_name },
              { label: 'Patient ID',         value: m.patient_id },
              { label: 'Date of Birth',      value: fmtDate(m.patient_dob) },
              { label: 'Sex',                value: m.patient_sex },
              { label: 'Study Date',         value: fmtDate(m.study_date) },
              { label: 'Study Description',  value: m.study_description },
              { label: 'Modality',           value: m.modality },
              { label: 'Accession No.',      value: m.accession_number },
              { label: 'Institution',        value: m.institution },
              { label: 'Referring Physician',value: m.referring_physician },
            ].filter(r => r.value);
            return (
              <div className="mt-4 rounded border border-[#c4c5d7] bg-white shadow-sm print:mt-0">
                <div className="flex items-center gap-2 border-b border-[#c4c5d7] bg-[#f2f0fb] px-4 py-2">
                  <span className="material-symbols-outlined text-[18px] text-[#434655]">badge</span>
                  <h3 className="text-[14px] font-semibold text-[#2c2f3a]">Patient Information</h3>
                  <span className="ml-1 rounded bg-[#dce3f9] px-2 py-0.5 text-[11px] font-semibold text-[#0037b0]">DICOM</span>
                </div>
                <div className="grid grid-cols-2 gap-x-6 gap-y-2 px-4 py-3 sm:grid-cols-3">
                  {rows.map(r => (
                    <div key={r.label}>
                      <div className="text-[11px] font-semibold uppercase tracking-[0.1em] text-[#747686]">{r.label}</div>
                      <div className="mt-0.5 text-[13px] font-medium text-[#1a1b23] break-words">{r.value}</div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })()}

          {error ? (
            <div className="mt-4 rounded border border-[#f1b4ad] bg-[#fff2f0] px-4 py-3 text-[14px] font-medium text-[#ba1a1a]">{error}</div>
          ) : null}

          <div className="grid grid-cols-12 items-start gap-4 pt-4">
            <div className="col-span-12 flex flex-col gap-4 lg:col-span-5">
              <Panel title="Primary Finding" icon="biotech">
                <div className="flex flex-col items-center px-6 py-6 text-center">
                  {result ? (
                    <>
                      <div className="mb-2 flex items-center gap-2">
                        <div className={`h-3 w-3 rounded-full ${result.flagged ? 'bg-[#ba1a1a]' : 'bg-[#0f9d58]'}`} />
                        <span className={`text-[12px] font-semibold uppercase tracking-[0.14em] ${result.flagged ? 'text-[#ba1a1a]' : 'text-[#0f9d58]'}`}>
                          {result.flagged ? 'Attention Required' : 'Review Clear'}
                        </span>
                      </div>
                      <div className={`text-[32px] font-bold leading-none sm:text-[40px] ${result.flagged ? 'text-[#ba1a1a]' : 'text-[#0f9d58]'}`}>{result.prediction}</div>

                      <div className="mt-6 w-full rounded border border-[#c4c5d7] bg-[#ededf9] p-3 text-left">
                        <div className="mb-2 flex items-center justify-between text-[12px] font-semibold uppercase tracking-[0.14em] text-[#575e70]">
                          <span>Confidence Score</span>
                          <span className="font-mono-ui text-[14px] font-bold text-[#1a1b23]">{confidencePercent}%</span>
                        </div>
                        <div className="h-2 w-full overflow-hidden rounded bg-[#e2e1ed]">
                          <div className={`h-full rounded ${result.flagged ? 'bg-[#ba1a1a]' : 'bg-[#0f9d58]'}`} style={{ width: `${confidencePercent}%` }} />
                        </div>
                      </div>

                      <div className="mt-4 text-left w-full text-[13px] text-[#575e70]">
                        <div className="font-semibold text-[#1a1b23]">Source</div>
                        <div className="mt-1">{result.filename}</div>
                        <div className="mt-2 rounded bg-[#f9fafb] px-3 py-2 text-[#434655] text-[13px] break-words">
                          <span className="font-semibold">Flags: </span>
                          {result.flagged ? (
                            <span className="text-[#ba1a1a] font-semibold">Flagged for Review</span>
                          ) : (
                            <span className="text-[#747686]">None</span>
                          )}
                        </div>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="mb-2 flex items-center gap-2">
                        <div className="h-3 w-3 rounded-full bg-[#575e70]" />
                        <span className="text-[12px] font-semibold uppercase tracking-[0.14em] text-[#575e70]">Waiting for upload</span>
                      </div>
                      <div className="text-[28px] font-bold leading-none text-[#1a1b23] sm:text-[36px]">No result yet</div>
                      <div className="mt-4 max-w-md text-[14px] text-[#575e70]">
                        Select an image on the left, run the analysis, and the model output will appear here.
                      </div>
                    </>
                  )}
                </div>
              </Panel>

              <Panel title="Class Probabilities" icon="bar_chart">
                <div className="px-4 py-4">
                  {result ? (
                    <table className="w-full border-collapse text-left">
                      <tbody>
                        {probabilityRows.map((row) => (
                          <tr key={row.label} className="border-b border-[#c4c5d7]/50 last:border-0">
                            <td className="w-[140px] py-3 text-[14px] font-semibold text-[#1a1b23]">{row.label}</td>
                            <td className="w-full px-3 py-3">
                              <div className="flex h-4 w-full items-center overflow-hidden rounded-sm bg-[#e2e1ed]">
                                <div className={`h-full ${row.value === Math.max(...probabilityRows.map((item) => item.value)) ? 'bg-[#0037b0]' : 'bg-[#575e70]'}`} style={{ width: `${row.value * 100}%` }} />
                              </div>
                            </td>
                            <td className="w-[70px] py-3 text-right font-mono-ui text-[13px] text-[#575e70]">{(row.value * 100).toFixed(1)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div className="px-4 py-10 text-center text-[14px] text-[#575e70]">Probabilities will appear after you run the model.</div>
                  )}
                </div>
              </Panel>
            </div>

            <div className="col-span-12 lg:col-span-7">
              <section className="flex h-full flex-col overflow-hidden rounded border border-[#c4c5d7] bg-white shadow-sm">
                <header className="flex items-center justify-between border-b border-[#c4c5d7] bg-[#f2f0fb] px-4 py-3">
                  <h3 className="flex items-center gap-2 text-[16px] font-semibold text-[#2c2f3a]">
                    <span className="material-symbols-outlined text-[18px] text-[#434655]">layers</span>
                    Grad-CAM++ Activation Map
                  </h3>
                  {result ? (
                    <div className="flex gap-2 text-[#575e70]">
                      <IconButton ariaLabel="Zoom in">
                        <span className="material-symbols-outlined text-[18px]">zoom_in</span>
                      </IconButton>
                      <IconButton ariaLabel="Expand preview" onClick={() => setPreviewExpanded(true)}>
                        <span className="material-symbols-outlined text-[18px]">open_in_full</span>
                      </IconButton>
                    </div>
                  ) : null}
                </header>

                <div className="flex flex-1 flex-col p-4">
                  <div className="grid min-h-[300px] flex-1 grid-cols-2 gap-4">
                    <div className="flex flex-col gap-2">
                      <span className="text-center text-[12px] font-semibold uppercase tracking-[0.14em] text-[#575e70]">Original Scan</span>
                      <div className="relative flex flex-1 items-center justify-center overflow-hidden rounded border border-[#c4c5d7] bg-black">
                        {previewUrl ? (
                          <img src={previewUrl} alt="Original Scan" className="absolute inset-0 h-full w-full object-contain" />
                        ) : (
                          <div className="px-4 text-center text-[14px] text-white/70">Upload an image to preview it here.</div>
                        )}
                      </div>
                    </div>

                    <div className="flex flex-col gap-2">
                      <span className="text-center text-[12px] font-semibold uppercase tracking-[0.14em] text-[#ba1a1a]">Activation Heatmap</span>
                      <div className="relative flex flex-1 items-center justify-center overflow-hidden rounded border border-[#c4c5d7] bg-black">
                        {gradcamDataUrl ? (
                          <img src={gradcamDataUrl} alt="Heatmap Scan" className="absolute inset-0 h-full w-full object-contain mix-blend-screen" />
                        ) : (
                          <div className="px-4 text-center text-[14px] text-white/70">Run analysis to generate the Grad-CAM overlay.</div>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="mt-4 flex items-center justify-center gap-3">
                    <span className="text-[12px] font-semibold uppercase tracking-[0.14em] text-[#575e70]">Low</span>
                    <div className="h-2 w-[260px] rounded" style={{ background: 'linear-gradient(to right, #000080, #0000ff, #00ffff, #ffff00, #ff0000)' }} />
                    <span className="text-[12px] font-semibold uppercase tracking-[0.14em] text-[#575e70]">High Activation</span>
                  </div>

                  {/* Grad-CAM Interpretation Guide */}
                  <div className="mt-3 rounded border border-[#c4c5d7] bg-[#f9fafb] px-4 py-3">
                    <div className="flex items-start gap-2">
                      <span className="material-symbols-outlined mt-[1px] shrink-0 text-[16px] text-[#0037b0]">info</span>
                      <div>
                        <p className="text-[12px] font-semibold text-[#1a1b23]">How to read this heatmap</p>
                        <p className="mt-1 text-[12px] leading-relaxed text-[#575e70]">
                          The AI highlights the regions it focused on to make its prediction. <span className="font-semibold text-[#ba1a1a]">Red / yellow areas</span> are the most influential — they contributed most to the diagnosis. <span className="font-semibold text-[#0037b0]">Blue / dark areas</span> were less relevant. Use these hotspots to verify the AI's attention aligns with the suspicious tissue region on the original scan.
                        </p>
                        <p className="mt-1 text-[11px] text-[#747686] italic">This is an assistive tool — always apply clinical judgement.</p>
                      </div>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          </div>

          {previewExpanded && result ? (
            <div
              className="fixed inset-0 z-30 flex items-center justify-center bg-[#1a1b23]/70 p-4 backdrop-blur-sm"
              role="dialog"
              aria-modal="true"
              aria-label="Expanded preview"
              onClick={() => setPreviewExpanded(false)}
            >
              <div
                className="w-full max-w-6xl overflow-hidden rounded border border-[#c4c5d7] bg-white shadow-[0_24px_64px_rgba(17,24,39,0.28)]"
                onClick={(event) => event.stopPropagation()}
              >
                <div className="flex items-center justify-between border-b border-[#c4c5d7] bg-[#f2f0fb] px-4 py-3">
                  <h3 className="flex items-center gap-2 text-[16px] font-semibold text-[#2c2f3a]">
                    <span className="material-symbols-outlined text-[18px] text-[#434655]">layers</span>
                    Grad-CAM++ Activation Map
                  </h3>
                  <button
                    type="button"
                    aria-label="Close expanded preview"
                    onClick={() => setPreviewExpanded(false)}
                    className="rounded border border-[#c4c5d7] bg-white px-3 py-1.5 text-[13px] font-semibold text-[#575e70] transition hover:bg-[#f3f2fe]"
                  >
                    Close
                  </button>
                </div>

                <div className="p-4">
                  <div className="grid min-h-[560px] grid-cols-2 gap-4">
                    <div className="flex flex-col gap-2">
                      <span className="text-center text-[12px] font-semibold uppercase tracking-[0.14em] text-[#575e70]">Original Scan</span>
                      <div className="relative flex flex-1 items-center justify-center overflow-hidden rounded border border-[#c4c5d7] bg-black">
                        <img src={previewUrl} alt="Original Scan enlarged" className="absolute inset-0 h-full w-full object-contain" />
                      </div>
                    </div>

                    <div className="flex flex-col gap-2">
                      <span className="text-center text-[12px] font-semibold uppercase tracking-[0.14em] text-[#ba1a1a]">Activation Heatmap</span>
                      <div className="relative flex flex-1 items-center justify-center overflow-hidden rounded border border-[#c4c5d7] bg-black">
                        <img src={gradcamDataUrl} alt="Heatmap enlarged" className="absolute inset-0 h-full w-full object-contain mix-blend-screen" />
                      </div>
                    </div>
                  </div>

                  <div className="mt-4 flex items-center justify-center gap-3">
                    <span className="text-[12px] font-semibold uppercase tracking-[0.14em] text-[#575e70]">Low</span>
                    <div className="h-2 w-[260px] rounded" style={{ background: 'linear-gradient(to right, #000080, #0000ff, #00ffff, #ffff00, #ff0000)' }} />
                    <span className="text-[12px] font-semibold uppercase tracking-[0.14em] text-[#575e70]">High Activation</span>
                  </div>

                  {/* Grad-CAM Interpretation Guide (expanded modal) */}
                  <div className="mt-3 rounded border border-[#c4c5d7] bg-[#f9fafb] px-4 py-3">
                    <div className="flex items-start gap-2">
                      <span className="material-symbols-outlined mt-[1px] shrink-0 text-[16px] text-[#0037b0]">info</span>
                      <div>
                        <p className="text-[12px] font-semibold text-[#1a1b23]">How to read this heatmap</p>
                        <p className="mt-1 text-[12px] leading-relaxed text-[#575e70]">
                          The AI highlights the regions it focused on to make its prediction. <span className="font-semibold text-[#ba1a1a]">Red / yellow areas</span> are the most influential — they contributed most to the diagnosis. <span className="font-semibold text-[#0037b0]">Blue / dark areas</span> were less relevant. Use these hotspots to verify the AI's attention aligns with the suspicious tissue region on the original scan.
                        </p>
                        <p className="mt-1 text-[11px] text-[#747686] italic">This is an assistive tool — always apply clinical judgement.</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : null}
        </main>
      </div>
    </div>
  );
}

export default Dashboard;