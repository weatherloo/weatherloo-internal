import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { useFonts } from "expo-font";
import { Inter_200ExtraLight, Inter_300Light, Inter_400Regular, Inter_500Medium, Inter_600SemiBold } from "@expo-google-fonts/inter";
import { View } from "react-native";
import { ForecastProvider } from "@/data/ForecastProvider";
import { nightTheme } from "@/theme/tokens";

/**
 * Root layout. Each screen is its own route file in src/app/ (e.g. add `globe.tsx`
 * for the AutoML globe); providers shared by all screens live here.
 */
export default function RootLayout() {
  const [loaded] = useFonts({
    Inter_200ExtraLight, Inter_300Light, Inter_400Regular, Inter_500Medium, Inter_600SemiBold,
  });
  if (!loaded) return <View style={{ flex: 1, backgroundColor: nightTheme.sky[0] }} />;
  return (
    <ForecastProvider>
      <StatusBar style="light" />
      <Stack screenOptions={{ headerShown: false, contentStyle: { backgroundColor: nightTheme.sky[0] } }} />
    </ForecastProvider>
  );
}
