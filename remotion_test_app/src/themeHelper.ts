import themeConfig from './themeConfig.json';

export interface ThemeConfig {
  background: string;
  accentColor: string;
  accentColorRgb: string;
  textColor: string;
  cardBackground: string;
  cardBorder: string;
  fontFamily: string;
  stiffness: number;
  damping: number;
  mass: number;
}

const OLD_THEME_MAP: Record<string, string> = {
  indigo: 'Civics',
  gold: 'History',
  emerald: 'Science',
  rose: 'Math'
};

export function getTheme(themeName: string): ThemeConfig {
  const mappedName = OLD_THEME_MAP[themeName.toLowerCase()] || themeName;
  const config = (themeConfig.themes as any)[mappedName];
  if (config) {
    return config;
  }
  return themeConfig.themes.General;
}
