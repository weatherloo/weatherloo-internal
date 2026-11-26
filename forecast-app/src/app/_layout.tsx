import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { useFonts } from "expo-font";
import { Newsreader_300Light, Newsreader_400Regular, Newsreader_400Regular_Italic } from "@expo-google-fonts/newsreader";
import { Inter_400Regular, Inter_500Medium, Inter_600SemiBold } from "@expo-google-fonts/inter";
import { View } from "react-native";
import { ForecastProvider } from "@/data/ForecastProvider";
import { colors } from "@/theme/tokens";

/**
 * Root layout. Each screen is its own route file in src/app/ (e.g. add `globe.tsx`
 * for the AutoML globe); providers shared by all screens live here.
 */
export default function RootLayout() {
  const [loaded] = useFonts({
    Newsreader_300Light, Newsreader_400Regular, Newsreader_400Regular_Italic,
    Inter_400Regular, Inter_500Medium, Inter_600SemiBold,
  });
  if (!loaded) return <View style={{ flex: 1, backgroundColor: colors.bg }} />;
  return (
    <ForecastProvider>
      <StatusBar style="light" />
      <Stack screenOptions={{ headerShown: false, contentStyle: { backgroundColor: colors.bg } }} />
    </ForecastProvider>
  );
}
