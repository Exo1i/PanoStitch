import 'react-native-gesture-handler';
import React from 'react';
import { NavigationContainer, DefaultTheme } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import PermissionScreen from './src/screens/PermissionScreen';
import HomeScreen from './src/screens/HomeScreen';
import SettingsScreen from './src/screens/SettingsScreen';
import StatusScreen from './src/screens/StatusScreen';
import PreviewScreen from './src/screens/PreviewScreen';

const colors = {
  background: '#0F172A',
  surface: '#1E293B',
  text: '#F8FAFC',
  textSecondary: '#94A3B8',
  primary: '#6366F1',
};

const AppTheme = {
  ...DefaultTheme,
  dark: true,
  colors: {
    ...DefaultTheme.colors,
    primary: colors.primary,
    background: colors.background,
    card: colors.surface,
    text: colors.text,
    border: colors.surface,
    notification: colors.primary,
  },
};

export type RootStackParamList = {
  Permission: undefined;
  Main: undefined;
  Settings: undefined;
  Status: undefined;
  Preview: { filename: string };
};

const Stack = createStackNavigator<RootStackParamList>();

export default function App() {
  return (
    <SafeAreaProvider>
      <NavigationContainer theme={AppTheme}>
        <StatusBar style="light" />
        <Stack.Navigator 
          initialRouteName="Permission"
          screenOptions={{
            headerStyle: { 
              backgroundColor: colors.surface,
              elevation: 0,
              shadowOpacity: 0,
              borderBottomWidth: 0,
            },
            headerTintColor: colors.text,
            headerTitleStyle: { 
              fontWeight: '600',
              fontSize: 18,
            },
            headerBackTitleVisible: false,
            cardStyle: { backgroundColor: colors.background },
          }}
        >
          <Stack.Screen 
            name="Permission" 
            component={PermissionScreen} 
            options={{ headerShown: false }} 
          />
          <Stack.Screen 
            name="Main" 
            component={HomeScreen} 
            options={{ headerShown: false }} 
          />
          <Stack.Screen 
            name="Settings" 
            component={SettingsScreen} 
            options={{ title: 'Settings' }} 
          />
          <Stack.Screen 
            name="Status" 
            component={StatusScreen} 
            options={{ 
              title: 'Processing',
              headerLeft: () => null,
              gestureEnabled: false,
            }} 
          />
          <Stack.Screen 
            name="Preview" 
            component={PreviewScreen} 
            options={{ headerShown: false }} 
          />
        </Stack.Navigator>
      </NavigationContainer>
    </SafeAreaProvider>
  );
}
