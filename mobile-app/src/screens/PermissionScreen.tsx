import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Linking, Alert, ActivityIndicator } from 'react-native';
import { useCameraPermissions } from 'expo-camera';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, typography } from '../utils/theme';

type RootStackParamList = {
  Main: undefined;
};

type NavigationProp = StackNavigationProp<RootStackParamList, 'Main'>;

export default function PermissionScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const navigation = useNavigation<NavigationProp>();

  React.useEffect(() => {
    if (permission?.granted) {
      navigation.replace('Main');
    }
  }, [permission, navigation]);

  const handleGrant = async () => {
    if (permission?.status === 'denied' && !permission.canAskAgain) {
      Alert.alert(
        "Camera Access Required",
        "PanoStitch needs camera access to capture panoramic photos. Please enable it in your device settings.",
        [
          { text: "Cancel", style: "cancel" },
          { text: "Open Settings", onPress: () => Linking.openSettings() }
        ]
      );
    } else {
      const result = await requestPermission();
      if (result.granted) {
        navigation.replace('Main');
      }
    }
  };

  if (!permission) {
    return (
      <View style={[styles.container, styles.centered]}>
        <ActivityIndicator size="large" color={colors.primary} />
        <Text style={styles.loadingText}>Loading...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Hero Icon */}
      <View style={styles.iconContainer}>
        <View style={styles.iconBox}>
          <Ionicons name="camera" size={48} color={colors.text} />
        </View>
      </View>

      {/* Title */}
      <Text style={styles.title}>PanoStitch</Text>
      <Text style={styles.subtitle}>Create stunning panoramas</Text>

      {/* Features */}
      <View style={styles.featureList}>
        <View style={styles.featureItem}>
          <Ionicons name="images-outline" size={20} color={colors.accent} />
          <Text style={styles.featureText}>Capture multiple shots</Text>
        </View>
        <View style={styles.featureItem}>
          <Ionicons name="git-merge-outline" size={20} color={colors.accent} />
          <Text style={styles.featureText}>AI-powered stitching</Text>
        </View>
        <View style={styles.featureItem}>
          <Ionicons name="compass-outline" size={20} color={colors.accent} />
          <Text style={styles.featureText}>Gyroscope preview</Text>
        </View>
      </View>

      {/* Permission Card */}
      <View style={styles.permissionCard}>
        <Ionicons name="shield-checkmark-outline" size={24} color={colors.textSecondary} />
        <Text style={styles.permissionText}>
          We need camera access to capture photos for your panoramas
        </Text>
      </View>

      {/* CTA Button */}
      <TouchableOpacity style={styles.button} onPress={handleGrant} activeOpacity={0.8}>
        <Text style={styles.buttonText}>Enable Camera</Text>
        <Ionicons name="arrow-forward" size={20} color={colors.text} />
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.xl,
    backgroundColor: colors.background,
  },
  centered: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    ...typography.body,
    color: colors.textSecondary,
    marginTop: spacing.md,
  },
  iconContainer: {
    marginBottom: spacing.lg,
  },
  iconBox: {
    width: 100,
    height: 100,
    borderRadius: borderRadius.xl,
    backgroundColor: colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  title: {
    ...typography.h1,
    color: colors.text,
    marginBottom: spacing.xs,
  },
  subtitle: {
    ...typography.body,
    color: colors.textSecondary,
    marginBottom: spacing.xxl,
  },
  featureList: {
    alignSelf: 'stretch',
    marginBottom: spacing.xxl,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  featureText: {
    ...typography.body,
    color: colors.text,
    marginLeft: spacing.md,
  },
  permissionCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.surfaceLight,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    marginBottom: spacing.xl,
  },
  permissionText: {
    ...typography.caption,
    color: colors.textSecondary,
    marginLeft: spacing.sm,
    flex: 1,
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.xl,
    borderRadius: borderRadius.full,
    backgroundColor: colors.primary,
    gap: spacing.sm,
  },
  buttonText: {
    ...typography.bodyBold,
    color: colors.text,
  },
});
