import { useMemo, useState } from 'react';
import type { KeyboardEvent, ReactNode } from 'react';
import Dashboard from './Dashboard';

type AuthView = 'create-account' | 'sign-in';
type AppView = 'auth' | 'dashboard';

const logoUrl =
  'https://lh3.googleusercontent.com/aida-public/AB6AXuDcwAXTnkJLapyxW105pHS_y8Z1HmakRh-rPHCc26mqqYFG8z1_c6oamw2yK2K0DarSyx647JIpwVvNf6a-UBeUNKViQ1-x2eZrLyzDaa7JpCJeQfYFncrTIi9rgR4lU_51RA-ctYeS76vPvgYTqi2llalvz1zuMHmWJ1O3SXWcY9hwkyTpf266Ys2S94ksagsYgRJi_BAIvuHLVIUSTeA0QcT_5LttLA-F6XjcFJbeunfSNQEI0hRN_kkXXe6rYAfxNA5fB6h60Rg';

function PageShell({ children }: { children: ReactNode }) {
  return <div className="grid min-h-screen place-items-center bg-[#faf8ff] p-6 text-[#1a1b23] antialiased sm:p-8">{children}</div>;
}

function OuterFooter() {
  return <div className="mt-6 text-center font-mono-ui text-[13px] text-[#747686]">v2.1.4 - BIDSS Clinical Module - © 2025</div>;
}

function FooterBar() {
  return <div className="border-t border-[#E5E7EB] bg-[#faf8ff] px-8 py-4 text-center text-[12px] text-[#747686]">v2.1.4 - BIDSS Clinical Module - © 2025</div>;
}

function AuthCard({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`w-full overflow-hidden rounded-[8px] border border-[#c4c5d7] bg-white shadow-[0_1px_3px_rgba(0,0,0,0.08),0_16px_36px_rgba(17,24,39,0.08)] ${className}`.trim()}>{children}</section>;
}

function Header({
  title,
  showTitle = true,
  logoSize = 'h-12',
  divider = true,
}: {
  title?: string;
  showTitle?: boolean;
  logoSize?: string;
  divider?: boolean;
}) {
  return (
    <div className={`bg-[#f9fafb] px-8 py-6 text-center ${divider ? 'border-b border-[#c4c5d7]' : ''}`}>
      <img src={logoUrl} alt="BIDSS Logo" className={`mx-auto ${logoSize} w-auto select-none object-contain`} draggable="false" />
      {showTitle && title ? <h1 className="mt-2 text-[18px] font-semibold leading-7 text-[#0037b0]">{title}</h1> : null}
    </div>
  );
}

function Field({
  label,
  id,
  type,
  value,
  placeholder,
  onChange,
  onKeyDown,
}: {
  label: string;
  id: string;
  type: string;
  value: string;
  placeholder?: string;
  onChange: (value: string) => void;
  onKeyDown?: (event: KeyboardEvent<HTMLInputElement>) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={id} className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[#434655]">
        {label}
      </label>
      <input
        id={id}
        type={type}
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={onKeyDown}
        aria-label={label}
        className="h-[42px] w-full rounded-md border border-[#c4c5d7] bg-white px-3 text-[14px] text-[#1a1b23] outline-none transition-all duration-200 placeholder:text-[#747686] focus:border-[#1d4ed8] focus:ring-1 focus:ring-[#1d4ed8]"
      />
    </div>
  );
}

function SelectField({
  label,
  id,
  value,
  options,
  onChange,
}: {
  label: string;
  id: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={id} className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[#434655]">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        aria-label={label}
        className="h-[42px] w-full rounded-md border border-[#c4c5d7] bg-white px-3 text-[14px] text-[#1a1b23] outline-none transition-all duration-200 focus:border-[#1d4ed8] focus:ring-1 focus:ring-[#1d4ed8]"
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  );
}

function App() {
  const [appView, setAppView] = useState<AppView>('auth');
  const [view, setView] = useState<AuthView>('create-account');

  const [loginEmail, setLoginEmail] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);
  const [loginBusy, setLoginBusy] = useState(false);
  const [loginSuccess, setLoginSuccess] = useState(false);

  const [title, setTitle] = useState('Dr.');
  const [fullName, setFullName] = useState('');
  const [designation, setDesignation] = useState('');
  const [signupEmail, setSignupEmail] = useState('');
  const [signupPassword, setSignupPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [signupError, setSignupError] = useState<string | null>(null);
  const [signupBusy, setSignupBusy] = useState(false);
  const [signupSuccess, setSignupSuccess] = useState(false);

  const loginCanSubmit = useMemo(() => loginEmail.trim().length > 0 && loginPassword.trim().length > 0 && !loginBusy, [loginBusy, loginEmail, loginPassword]);
  const signupCanSubmit = useMemo(
    () =>
      title.trim().length > 0 &&
      fullName.trim().length > 0 &&
      designation.trim().length > 0 &&
      signupEmail.trim().length > 0 &&
      signupPassword.trim().length > 0 &&
      confirmPassword.trim().length > 0 &&
      !signupBusy,
    [confirmPassword, designation, fullName, signupBusy, signupEmail, signupPassword, title],
  );

  const submitLogin = () => {
    if (!loginEmail.trim() || !loginPassword.trim()) {
      setLoginError('Please enter your credentials.');
      return;
    }

    setLoginError(null);
    setLoginBusy(true);
    setLoginSuccess(false);

    window.setTimeout(() => {
      setLoginSuccess(true);
      window.setTimeout(() => {
        setLoginBusy(false);
        setLoginSuccess(false);
        setLoginEmail('');
        setLoginPassword('');
        setRememberMe(false);
        setAppView('dashboard');
      }, 1800);
    }, 900);
  };

  const submitSignup = () => {
    if (!title.trim() || !fullName.trim() || !designation.trim() || !signupEmail.trim() || !signupPassword.trim() || !confirmPassword.trim()) {
      setSignupError('Please fill in all fields.');
      return;
    }

    if (signupPassword !== confirmPassword) {
      setSignupError('Passwords do not match.');
      return;
    }

    setSignupError(null);
    setSignupBusy(true);
    setSignupSuccess(false);

    window.setTimeout(() => {
      setSignupSuccess(true);
      window.setTimeout(() => {
        setSignupBusy(false);
        setSignupSuccess(false);
        setTitle('Dr.');
        setFullName('');
        setDesignation('');
        setSignupEmail('');
        setSignupPassword('');
        setConfirmPassword('');
      }, 1800);
    }, 900);
  };

  if (appView === 'dashboard') {
    return <Dashboard onSignOut={() => setAppView('auth')} />;
  }

  return (
    <PageShell>
      {view === 'create-account' ? (
        <>
          <AuthCard className="max-w-[360px]">
            <Header title="Create Account" logoSize="h-20" />

            <div className="px-8 py-6">
              <div className="space-y-4">
                <div className="flex gap-4">
                  <div className="w-1/3">
                    <SelectField label="Title" id="title" value={title} options={['Dr.', 'Prof.', 'Mr.', 'Ms.']} onChange={setTitle} />
                  </div>
                  <div className="w-2/3">
                    <Field label="Full Name" id="fullName" type="text" value={fullName} placeholder="e.g. Jane Doe" onChange={setFullName} />
                  </div>
                </div>

                <Field label="Designation" id="designation" type="text" value={designation} placeholder="e.g. Radiologist" onChange={setDesignation} />
                <Field label="Email Address" id="signupEmail" type="email" value={signupEmail} placeholder="jane.doe@clinic.com" onChange={setSignupEmail} />
                <Field label="Password" id="signupPassword" type="password" value={signupPassword} onChange={setSignupPassword} />
                <Field
                  label="Confirm Password"
                  id="confirmPassword"
                  type="password"
                  value={confirmPassword}
                  onChange={setConfirmPassword}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault();
                      submitSignup();
                    }
                  }}
                />

                <div className="pt-1">
                  <button
                    type="button"
                    aria-label="Create Account"
                    onClick={submitSignup}
                    disabled={!signupCanSubmit}
                    className="flex h-10 w-full items-center justify-center rounded-md bg-[#1d4ed8] text-[14px] font-semibold text-white transition-all duration-200 hover:bg-[#0037b0] disabled:cursor-not-allowed disabled:opacity-70"
                  >
                    {signupBusy ? (
                      <span className="inline-flex items-center gap-2">
                        <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" aria-hidden="true" />
                        Creating Account
                      </span>
                    ) : signupSuccess ? (
                      'Account Created'
                    ) : (
                      'Create Account'
                    )}
                  </button>
                </div>

                {signupError ? (
                  <p className="pt-1 text-center text-[14px] font-medium text-[#ba1a1a]" role="alert">
                    {signupError}
                  </p>
                ) : null}

                <button
                  type="button"
                  aria-label="Already have an account? Sign In"
                  onClick={() => setView('sign-in')}
                  className="block w-full text-center text-[14px] font-semibold text-[#0037b0] transition hover:underline"
                >
                  Already have an account? Sign In
                </button>
              </div>
            </div>
          </AuthCard>

          <OuterFooter />
        </>
      ) : (
        <AuthCard className="max-w-[380px]">
          <Header showTitle={false} logoSize="h-20" divider={false} />

          <div className="px-8 py-6">
            <div className="space-y-5">
              <Field label="Email Address" id="loginEmail" type="email" value={loginEmail} placeholder="Enter your email" onChange={setLoginEmail} />
              <Field
                label="Password"
                id="loginPassword"
                type="password"
                value={loginPassword}
                placeholder="Enter password"
                onChange={setLoginPassword}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault();
                    submitLogin();
                  }
                }}
              />

              <label className="mt-1 flex items-center gap-2 text-[13px] text-[#434655]">
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(event) => setRememberMe(event.target.checked)}
                  aria-label="Remember me on this device"
                  className="h-4 w-4 border-[#c4c5d7] text-[#1d4ed8] focus:ring-[#1d4ed8]/20"
                />
                <span>Remember me on this device</span>
              </label>

              <div className="space-y-3 pt-2">
                <button
                  type="button"
                  aria-label="Sign In"
                  onClick={submitLogin}
                  disabled={!loginCanSubmit}
                    className="flex h-10 w-full items-center justify-center rounded-md bg-[#1d4ed8] text-[14px] font-semibold text-white transition-all duration-200 hover:bg-[#0037b0] disabled:cursor-not-allowed disabled:opacity-70"
                >
                  {loginBusy ? (
                    <span className="inline-flex items-center gap-2">
                      <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" aria-hidden="true" />
                      Signing In
                    </span>
                  ) : loginSuccess ? (
                    'Success'
                  ) : (
                    'Sign In'
                  )}
                </button>

                <button
                  type="button"
                  aria-label="Create an Account"
                  onClick={() => setView('create-account')}
                    className="flex h-10 w-full items-center justify-center rounded-md border border-[#c4c5d7] bg-white text-[14px] font-semibold text-[#1a1b23] transition-all duration-200 hover:bg-[#ededf9]"
                >
                  Create an Account
                </button>
              </div>

              {loginError ? (
                <p className="pt-1 text-center text-[14px] font-medium text-[#ba1a1a]" role="alert">
                  {loginError}
                </p>
              ) : null}

              <div className="pt-3 text-center">
                <p className="text-[12px] text-[#747686]">For IT support contact ext. 4400</p>
              </div>
            </div>
          </div>

          <FooterBar />
        </AuthCard>
      )}
    </PageShell>
  );
}

export default App;