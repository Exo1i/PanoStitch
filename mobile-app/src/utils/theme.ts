export const colors = {
  primary: '#6366F1',       // Indigo
  primaryDark: '#4F46E5',
  primaryLight: '#818CF8',
  
  accent: '#22D3EE',        // Cyan
  accentDark: '#06B6D4',
  
  background: '#0F172A',    // Slate 900
  surface: '#1E293B',       // Slate 800
  surfaceLight: '#334155',  // Slate 700
  
  text: '#F8FAFC',          // Slate 50
  textSecondary: '#94A3B8', // Slate 400
  textMuted: '#64748B',     // Slate 500
  
  success: '#22C55E',
  error: '#EF4444',
  warning: '#F59E0B',
  
  overlay: 'rgba(15, 23, 42, 0.8)',
  overlayLight: 'rgba(30, 41, 59, 0.9)',
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

export const borderRadius = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  full: 999,
};

export const typography = {
  h1: { fontSize: 32, fontWeight: '700' as const, letterSpacing: -0.5 },
  h2: { fontSize: 24, fontWeight: '600' as const, letterSpacing: -0.3 },
  h3: { fontSize: 20, fontWeight: '600' as const },
  body: { fontSize: 16, fontWeight: '400' as const },
  bodyBold: { fontSize: 16, fontWeight: '600' as const },
  caption: { fontSize: 14, fontWeight: '400' as const },
  small: { fontSize: 12, fontWeight: '400' as const },
};
