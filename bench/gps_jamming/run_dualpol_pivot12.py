"""
Pivot 1 + Pivot 2 : Dual-Pol (RHCP/LHCP) sedenion jammer excision + passive S-parameter stage + CRLB.
numpy only.  Every number printed is computed here.
"""
import numpy as np, json, sys
np.set_printoptions(precision=4, suppress=True, linewidth=140)
rng = np.random.default_rng(20261025)

# ---------------- Cayley-Dickson sedenions ----------------
def cd_mul(a, b):
    n = len(a)
    if n == 1: return np.array([a[0]*b[0]])
    h = n//2; a1,a2,b1,b2 = a[:h],a[h:],b[:h],b[h:]
    def conj(x): c=-x.copy(); c[0]=x[0]; return c
    return np.concatenate([cd_mul(a1,b1)-cd_mul(conj(b2),a2), cd_mul(b2,a1)+cd_mul(a2,conj(b1))])
T = np.zeros((16,16,16))
for i in range(16):
    for j in range(16):
        ei=np.zeros(16); ei[i]=1; ej=np.zeros(16); ej[j]=1
        T[:,i,j]=cd_mul(ei,ej)
L = lambda u: np.einsum('kij,i->kj', T, u)     # (u*x) = L(u) x
R = lambda v: np.einsum('kij,j->ki', T, v)     # (x*v) = R(v) x
def sconj(v): c=-v.copy(); c[0]=v[0]; return c
def omul(a,b): return cd_mul(a,b)
def oconj(a): c=-a.copy(); c[0]=a[0]; return c
db = lambda p: 10*np.log10(np.maximum(p,1e-300))

report = {}

# ================= 1. PIVOT 1: passive S-matrix of the zero-divisor stage =================
print("=== 1. PIVOT 1: passive reciprocal 32-port S-matrix, S21 = R(v)/sqrt2 =================")
def smatrix_facts(v, label):
    v = v/np.linalg.norm(v)
    S21 = R(v)/np.sqrt(2)
    sv = np.linalg.svd(S21, compute_uv=False)
    S32 = np.block([[np.zeros((16,16)), S21.T],[S21, np.zeros((16,16))]])
    passive = np.linalg.eigvalsh(np.eye(32)-S32.T@S32).min()
    sym = np.abs(S32-S32.T).max()
    # reciprocity through the adjoint identity R(v)^T = R(conj v)
    adj = np.abs(R(v).T - R(sconj(v))).max()
    nz = (np.abs(S21)>1e-12).sum(axis=1)
    vals = sorted(set(np.round(np.abs(S21[np.abs(S21)>1e-12]),6)))
    print(f"{label}: sing.values(S21) = {np.round(sv,4)}")
    print(f"   min eig(I - S32^T S32) = {passive:+.2e} (>=0 passive), ||S32-S32^T||_max = {sym:.1e}, ||R(v)^T - R(v*)||_max = {adj:.1e}")
    print(f"   couplings per output port = {sorted(set(nz))}, coupling magnitudes = {vals}")
    return dict(singular_values=np.round(sv,6).tolist(), min_eig_passivity=float(passive), symmetry_err=float(sym),
                adjoint_err=float(adj), couplings_per_port=sorted(set(int(x) for x in nz)), coupling_mags=[float(x) for x in vals])
u_can = np.zeros(16); u_can[3]=1; u_can[12]=1         # e3 + e12 (ternary Moreno blade)
report['smatrix_canonical'] = smatrix_facts(u_can, "canonical blade (e3+e12)/sqrt2")
# a general-position Moreno annihilator v(c) = (c, (ac)b)
def rand_cone():
    a=rng.standard_normal(8); a[0]=0; a/=np.linalg.norm(a)
    b=rng.standard_normal(8); b[0]=0; b-=a*(a@b); b/=np.linalg.norm(b); return a,b
def fibre_basis(a,b):
    ab=omul(a,b); B=np.stack([np.eye(8)[0],a,b,ab]).T
    Q,_=np.linalg.qr(np.concatenate([B, rng.standard_normal((8,4))],1)); return Q[:,4:]
def annihilator(a,b,c): return np.concatenate([c, omul(omul(a,c),b)])
a,b=rand_cone(); c=fibre_basis(a,b)@rng.standard_normal(4); v_gen=annihilator(a,b,c)
report['smatrix_general'] = smatrix_facts(v_gen, "general-position v(c)=(c,(ac)b)")
print(f"   check (a,b).v(c) = 0 : ||.|| = {np.linalg.norm(cd_mul(np.concatenate([a,b]),v_gen)):.2e};  two-sided v(c).(a,b): {np.linalg.norm(cd_mul(v_gen,np.concatenate([a,b]))):.2e}")

# ================= 2. PIVOT 2: dual-pol embedding, gauge-free (T = I) =================
print("\n=== 2. PIVOT 2: dual-pol embedding (fixed wiring, T = I):  RHCP I/Q -> (e1,e2),  LHCP I/Q -> (e5,e6) ===")
# complex RHCP sample z -> a = Re z e1 + Im z e2 ; complex LHCP sample w -> b = Re w e4 + Im w e5
E_R = np.zeros((8,2)); E_R[1,0]=1; E_R[2,1]=1
E_L = np.zeros((8,2)); E_L[5,0]=1; E_L[6,1]=1
def embed(z, w):
    """complex RHCP z, LHCP w  ->  sedenion (a,b) in R^16 (identity gauge)"""
    return np.concatenate([E_R@np.array([z.real,z.imag]), E_L@np.array([w.real,w.imag])])

def pol_vector(kind, psi_deg=0.0, axial_ratio_db=np.inf):
    """Jones vector in the (RHCP, LHCP) circular basis, unit power.
       linear at tilt psi: (e^{-j psi}, e^{+j psi})/sqrt2 ; elliptical with axial ratio AR (dB): amplitudes unequal."""
    psi=np.deg2rad(psi_deg)
    if kind=='RHCP': return np.array([1,0],complex)
    if kind=='LHCP': return np.array([0,1],complex)
    if kind=='linear':
        return np.array([np.exp(-1j*psi), np.exp(1j*psi)])/np.sqrt(2)
    if kind=='elliptical':
        AR=10**(axial_ratio_db/20)            # AR = |E_major|/|E_minor|;  circular amps: (AR+1)/2, (AR-1)/2 (normalised)
        r,l=(AR+1)/2,(AR-1)/2; n=np.hypot(r,l)
        return np.array([r*np.exp(-1j*psi), l*np.exp(1j*psi)])/n

# the jammer signal subspace in R^16: X_J(t) = embed(s(t) pJ[0], s(t) pJ[1]) for complex s(t) -> 2-D real subspace
def signal_subspace(p):
    return np.stack([embed(p[0]*s, p[1]*s) for s in (1.0, 1j)], axis=1)   # 16x2
pJ = pol_vector('linear', psi_deg=37.0)
UJ = signal_subspace(pJ)
# cone check per sample
for s in (1.0, 1j, np.exp(1j*0.7)):
    X=embed(pJ[0]*s,pJ[1]*s); a_,b_=X[:8],X[8:]
    print(f"   sample s={s:.3f}: |a|={np.linalg.norm(a_):.4f} |b|={np.linalg.norm(b_):.4f} <a,b>={a_@b_:.1e}  -> on Moreno cone")
# common annihilator for the whole 2-D jammer subspace: intersect ker L(X_I) and ker L(X_Q)
def common_right_kernel(U):
    """v with X.v = 0 for all X in col(U): null space of stacked L(X_k)"""
    M=np.concatenate([L(U[:,k]/np.linalg.norm(U[:,k])) for k in range(U.shape[1])],0)
    _,s,Vt=np.linalg.svd(M); return Vt[(s<1e-9).sum()*0 + (s>1e-9).sum():].T, s
K, s_stack = common_right_kernel(UJ)
print(f"   dim ker L(X_I) = {(np.linalg.svd(L(UJ[:,0]),compute_uv=False)<1e-9).sum()},  dim common kernel over the I/Q jammer subspace = {K.shape[1]}")
report['common_kernel_dim']=int(K.shape[1])
# closed-form tilt-dependent annihilator (Lean: dualpol_tilt_annihilation):  v(ψ) = e3 + sin(2ψ) e12 - cos(2ψ) e15
psi=np.deg2rad(37.0)
v=np.zeros(16); v[3]=1; v[12]=np.sin(2*psi); v[15]=-np.cos(2*psi); v/=np.linalg.norm(v)
XI=UJ[:,0]/np.linalg.norm(UJ[:,0]); XQ=UJ[:,1]/np.linalg.norm(UJ[:,1])
print(f"   closed-form v(ψ)=e3+sin2ψ e12-cos2ψ e15: ||X_I.v||={np.linalg.norm(cd_mul(XI,v)):.1e}  ||X_Q.v||={np.linalg.norm(cd_mul(XQ,v)):.1e}  (also two-sided: ||v.X_I||={np.linalg.norm(cd_mul(v,XI)):.1e})")
print(f"   v(ψ) is itself on the Moreno cone: |c|={np.linalg.norm(v[:8]):.4f} |d|={np.linalg.norm(v[8:]):.4f} <c,d>={v[:8]@v[8:]:.1e}; v lies in the {K.shape[1]}-D common kernel: residual {np.linalg.norm(K@(K.T@v)-v):.1e}")
# the general-position theorem annihilator for the I-sample (unit a, b) is a different element of ker L(X_I); it kills X_I but not X_Q:
aI,bI=XI[:8]/np.linalg.norm(XI[:8]),XI[8:]/np.linalg.norm(XI[8:]); cI=fibre_basis(aI,bI)[:,0]; v_pt=annihilator(aI,bI,cI); v_pt/=np.linalg.norm(v_pt)
print(f"   generic point-annihilator v(c) of X_I: ||X_I.v(c)||={np.linalg.norm(cd_mul(XI,v_pt)):.1e} but ||X_Q.v(c)||={np.linalg.norm(cd_mul(XQ,v_pt)):.3f} -> must pick v in the common kernel (the wiring makes it exist)")
# wiring sensitivity: wrong LHCP plane -> no common kernel
E_bad=np.zeros((8,2)); E_bad[4,0]=1; E_bad[5,1]=1
UJ_bad=np.stack([np.concatenate([E_R@np.array([(pJ[0]*s_).real,(pJ[0]*s_).imag]), E_bad@np.array([(pJ[1]*s_).real,(pJ[1]*s_).imag])]) for s_ in (1.0,1j)],1)
print(f"   wiring check: LHCP -> (e4,e5) instead gives common-kernel dim {common_right_kernel(UJ_bad)[0].shape[1]} (no polarisation-only null exists); (e5,e6) gives {K.shape[1]}")
report['wiring']={'e5e6_common_kernel_dim':int(K.shape[1]),'e4e5_common_kernel_dim':int(common_right_kernel(UJ_bad)[0].shape[1])}

# ---- Δθ = 0 scenario: satellite RHCP, same spatial direction, same complex waveform support ----
print("\n--- Δθ = 0°, polarisation-only separation; outputs through R(v), satellite read out by least squares ---")
def run_excision(v, pS, pJ, xpd_db=np.inf, n=4000, jnr_db=80.0, snr_db=0.0):
    """Monte-Carlo: X(t) = s(t) embed(pS) + j(t) embed(pJ) + noise. Return powers through R(v)."""
    s = (rng.standard_normal(n)+1j*rng.standard_normal(n))/np.sqrt(2)
    j = (rng.standard_normal(n)+1j*rng.standard_normal(n))/np.sqrt(2)*10**(jnr_db/20)
    # finite cross-pol discrimination of the receive antenna leaks satellite into LHCP port
    leak = 10**(-xpd_db/20)
    pS_eff = np.array([pS[0], pS[1]+leak*pS[0]]); pS_eff/=np.linalg.norm(pS_eff)
    XS = np.stack([embed(pS_eff[0]*si, pS_eff[1]*si) for si in s],1)
    XJ = np.stack([embed(pJ[0]*ji, pJ[1]*ji) for ji in j],1)
    N  = rng.standard_normal((16,n))*np.sqrt(10**(-snr_db/10)/4)   # per-sample real noise, total power matched to 4 active real dims
    N[[0,3,6,7,8,11,14,15],:]=0   # only the 4 physical real channels (I/Q x RHCP/LHCP) carry noise
    Rv = R(v)
    YS, YJ, YN = Rv@XS, Rv@XJ, Rv@N
    pin_S, pin_J = (XS**2).sum()/n, (XJ**2).sum()/n
    pout_S, pout_J = (YS**2).sum()/n, (YJ**2).sum()/n
    return dict(sat_in=pin_S, sat_out=pout_S, jam_in=pin_J, jam_out=pout_J, YS=YS, YJ=YJ, YN=YN, XS=XS, N=N, s=s)
pS = pol_vector('RHCP')
res = run_excision(v, pS, pJ)
print(f"   satellite: in {db(res['sat_in']):+.4f} dB  out {db(res['sat_out']):+.4f} dB  -> loss {db(res['sat_in']/res['sat_out']):.4f} dB")
print(f"   jammer   : in {db(res['jam_in']):+.1f} dB  out {db(res['jam_out']):+.1f} dB  -> suppression {db(res['jam_in']/res['jam_out']):.1f} dB (float64 floor)")
report['delta_theta_0'] = dict(sat_loss_db=float(db(res['sat_in']/res['sat_out'])), jam_suppression_db=float(db(res['jam_in']/res['jam_out'])))

# ---- fair baseline: dual-pol 2-port LCMV (distortionless to RHCP, hard null on pJ) & output SNR comparison ----
print("\n--- Fair baseline: conventional dual-pol 2-port LCMV (w^H pS = 1, w^H pJ = 0) vs sedenion stage + LS readout ---")
def lcmv_2port(pS,pJ):
    C=np.stack([pS,pJ],1); f=np.array([1,0],complex)
    return C@np.linalg.solve(C.conj().T@C, f)          # min-norm solution of C^H w = f
w = lcmv_2port(pS,pJ)
print(f"   LCMV: |w^H pS|^2 = {abs(np.vdot(w,pS))**2:.4f}, |w^H pJ|^2 = {abs(np.vdot(w,pJ))**2:.1e}, noise gain ||w||^2 = {np.linalg.norm(w)**2:.4f} ({db(np.linalg.norm(w)**2):.2f} dB SNR penalty)")
# sedenion: output Y = R(v) X ; recover complex s by LS on the satellite's image columns G = R(v) embed(pS)
G = R(v)@signal_subspace(pS)                          # 16x2 real, images of I and Q satellite samples
Wls = np.linalg.pinv(G)                               # 2x16 LS readout
sig_gain = np.linalg.norm(Wls@G,'fro')**2/2           # =1
noise_cov = (Wls@R(v))                                # noise passes through R(v) then readout; only 4 physical dims carry noise
phys = [1,2,4,5]
noise_gain = (noise_cov[:,phys]**2).sum()/2 / (1/2)    # relative to a matched filter on the 2 RHCP dims (each dim var 1/4 -> matched noise 1/2 per complex sample... normalised)
print(f"   sedenion+LS: signal gain {sig_gain:.4f}, jammer residual {np.linalg.norm(Wls@R(v)@UJ):.1e}, noise gain {noise_gain:.4f} ({db(noise_gain):.2f} dB SNR penalty)")
# Also: the best possible linear 16-D LCMV in the embedded real domain (oblique projector): same as 2-port
report['lcmv_noise_gain_db']=float(db(np.linalg.norm(w)**2)); report['sedenion_noise_gain_db']=float(db(noise_gain))

# ---- Δθ sweep: spatial MVDR (8-el ULA, single pol) vs dual-pol sedenion (angle-independent) ----
print("\n--- Δθ sweep: 8-element ULA spatial MVDR (RHCP only) vs dual-pol sedenion excision ---")
def ula(theta_deg,M=8):
    m=np.arange(M); return np.exp(1j*np.pi*m*np.sin(np.deg2rad(theta_deg)))/np.sqrt(M)
def spatial_mvdr_sinr(dtheta, jnr_db=80, snr_db=0, M=8):
    aS=ula(10.0,M); aJ=ula(10.0+dtheta,M)
    Rn = np.eye(M)*10**(-snr_db/10) + 10**(jnr_db/10)*np.outer(aJ,aJ.conj())
    wm = np.linalg.solve(Rn,aS); wm/=np.vdot(aS,wm)
    sig = abs(np.vdot(wm,aS))**2
    interf = 10**(jnr_db/10)*abs(np.vdot(wm,aJ))**2
    noise = 10**(-snr_db/10)*np.linalg.norm(wm)**2
    return db(sig/(interf+noise))
rows=[]
for dth in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]:
    sinr_sp = spatial_mvdr_sinr(dth)
    # sedenion: polarisation only -> independent of Δθ; output SINR = SNR - noise penalty (jammer at float floor)
    sinr_sed = 0.0 - db(noise_gain)
    sinr_lcmv = 0.0 - db(np.linalg.norm(w)**2)
    rows.append(dict(dtheta=dth, spatial_mvdr_sinr_db=float(sinr_sp), dualpol_sedenion_sinr_db=float(sinr_sed), dualpol_lcmv_sinr_db=float(sinr_lcmv)))
    print(f"   Δθ={dth:5.1f}°  spatial MVDR SINR {sinr_sp:+7.2f} dB | dual-pol sedenion {sinr_sed:+6.2f} dB | dual-pol LCMV {sinr_lcmv:+6.2f} dB   (input SNR 0 dB, J/N 80 dB)")
report['dtheta_sweep']=rows

# ---- robustness: jammer not perfectly linear (axial ratio), satellite XPD leakage, calibration error ----
print("\n--- Robustness: elliptical jammer (finite axial ratio) -> leaves the cone; leakage vs 2||δ||² bound ---")
rows=[]
for ar_db in [np.inf, 40, 30, 20, 10, 6, 3]:
    pJe = pol_vector('elliptical', psi_deg=37.0, axial_ratio_db=ar_db)
    UJe = signal_subspace(pJe); 
    # v designed for the linear (cone) signature pJ; true signature pJe = pJ + δ
    res_e = run_excision(v, pS, pJe, n=2000)
    sup = db(res_e['jam_in']/res_e['jam_out'])
    delta = np.linalg.norm(UJe-UJ,'fro')**2/2     # ||δ||² per unit-power sample (avg over I/Q)
    bound = db(1/(2*delta)) if delta>0 else np.inf
    rows.append(dict(axial_ratio_db=float(ar_db) if np.isfinite(ar_db) else None, suppression_db=float(sup), leakage_bound_db=float(bound) if np.isfinite(bound) else None))
    print(f"   jammer axial ratio {str(ar_db):>4} dB: suppression {sup:6.1f} dB   | leakage_bound  -10log10(2||δ||²) = {bound:6.1f} dB  (bound {'holds' if sup>=bound-1e-6 else 'VIOLATED'})")
report['axial_ratio_sweep']=rows
print("   (a jammer that is itself circularly polarised, AR -> 0 dB, is not on the cone and is not excised: this is the honest limit of Pivot 2)")
print("\n--- Satellite cross-pol discrimination (receive antenna XPD) -> satellite acquires a small LHCP component ---")
rows=[]
for xpd in [np.inf, 40, 30, 25, 20, 15]:
    r_ = run_excision(v, pS, pJ, xpd_db=xpd, n=2000)
    loss = db(r_['sat_in']/r_['sat_out'])
    rows.append(dict(xpd_db=float(xpd) if np.isfinite(xpd) else None, sat_loss_db=float(loss)))
    print(f"   XPD {str(xpd):>4} dB: satellite loss through R(v) = {loss:.4f} dB")
report['xpd_sweep']=rows
print("\n--- Weight calibration error δ_v (phase/amplitude error of the 16 couplers) ---")
rows=[]
for eps in [0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]:
    dv = rng.standard_normal(16); dv*=eps/np.linalg.norm(dv)
    r_ = run_excision(v+dv, pS, pJ, n=2000)
    sup = db(r_['jam_in']/r_['jam_out']); bound = db(1/(2*eps**2)) if eps>0 else np.inf
    rows.append(dict(eps=eps, suppression_db=float(sup), bound_db=float(bound) if np.isfinite(bound) else None))
    print(f"   ||δ_v||={eps:7.0e}: suppression {sup:6.1f} dB | bound -10log10(2||δ||²) = {bound:6.1f} dB")
report['calibration_sweep']=rows

# ================= 3. CRLB on code phase (pseudorange) =================
print("\n=== 3. CRLB: GPS L1 C/A code-phase (pseudorange) after excision =============================")
# Gabor-bandwidth CRLB: var(τ) >= 1 / ( 2 (C/N0)_eff T_int (2π)^2 β_rms^2 ),  β_rms^2 = ∫f²|S(f)|²/∫|S(f)|²
fc=1.023e6; Bfe=2.0e6                     # chip rate, front-end bandwidth (±1 MHz)
f=np.linspace(-Bfe,Bfe,200001); Sf=np.sinc(f/fc)**2; beta2=np.trapezoid(f**2*Sf,f)/np.trapezoid(Sf,f)
cN0_db=45.0; Tint=1e-3; c0=299792458.0
def crlb_range_m(cn0_eff_db):
    cn0=10**(cn0_eff_db/10); var=1/(2*cn0*Tint*(2*np.pi)**2*beta2); return c0*np.sqrt(var)
print(f"   β_rms = {np.sqrt(beta2)/1e3:.1f} kHz, T_int = {Tint*1e3:.0f} ms, nominal C/N0 = {cN0_db} dB-Hz -> CRLB σ_ρ = {crlb_range_m(cN0_db):.3f} m (no jammer)")
JN_db=80.0  # jammer-to-noise in the 2 MHz band
rows=[]
for dth in [0.0, 1.0, 5.0, 30.0]:
    # effective C/N0 = C/N0 - 10log10(1 + residual J/N) - noise penalty
    sp = spatial_mvdr_sinr(dth)                     # SINR at SNR 0 dB -> degradation = -sp
    eff_sp = cN0_db + sp
    eff_sed = cN0_db - db(noise_gain) - db(1+10**((JN_db-report['delta_theta_0']['jam_suppression_db'])/10))
    eff_lcmv = cN0_db - db(np.linalg.norm(w)**2)
    eff_none = cN0_db - db(1+10**(JN_db/10))
    row=dict(dtheta=dth, crlb_none_m=crlb_range_m(eff_none), crlb_spatial_mvdr_m=crlb_range_m(eff_sp), crlb_dualpol_sedenion_m=crlb_range_m(eff_sed), crlb_dualpol_lcmv_m=crlb_range_m(eff_lcmv))
    rows.append({k:(float(x) if isinstance(x,(float,np.floating)) else x) for k,x in row.items()})
    print(f"   Δθ={dth:5.1f}°  σ_ρ: no mitigation {row['crlb_none_m']:9.1f} m | spatial MVDR {row['crlb_spatial_mvdr_m']:8.3f} m | dual-pol sedenion {row['crlb_dualpol_sedenion_m']:.3f} m | dual-pol LCMV {row['crlb_dualpol_lcmv_m']:.3f} m")
report['crlb']=rows

json.dump(report, open(sys.argv[1] if len(sys.argv)>1 else 'results_dualpol_pivot12.json','w'), indent=1, default=float)
print("\nwrote results json")
