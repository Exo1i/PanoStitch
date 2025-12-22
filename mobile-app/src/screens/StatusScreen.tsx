import React from 'react';
import { View, Text, StyleSheet, ActivityIndicator, FlatList, Image, TouchableOpacity, Dimensions } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useAppStore } from '../store/useAppStore';
import { useNavigation } from '@react-navigation/native';
import { colors, spacing, borderRadius, typography } from '../utils/theme';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export default function StatusScreen() {
  const { jobStatus, panoramas, apiBaseUrl, currentSessionId, errorMessage, resetSession } = useAppStore();
  const navigation = useNavigation<any>();
  const insets = useSafeAreaInsets();

  const handlePreview = (filename: string) => {
    navigation.navigate('Preview', { filename });
  };

  const handleDone = () => {
    resetSession();
    navigation.navigate('Main');
  };

  if (jobStatus === 'uploading' || jobStatus === 'processing') {
    return (
      <View style={styles.centerContainer}>
        <View style={styles.loaderContainer}>
          <ActivityIndicator size="large" color={colors.primary} />
        </View>
        <Text style={styles.statusTitle}>
          {jobStatus === 'uploading' ? 'Uploading...' : 'Stitching...'}
        </Text>
        <Text style={styles.statusDescription}>
          {jobStatus === 'uploading' 
            ? 'Sending images to server' 
            : 'Creating your panorama'
          }
        </Text>
      </View>
    );
  }

  if (jobStatus === 'error') {
    return (
      <View style={styles.centerContainer}>
        <View style={styles.errorIcon}>
          <Ionicons name="alert-circle" size={48} color={colors.error} />
        </View>
        <Text style={styles.errorTitle}>Stitching Failed</Text>
        <Text style={styles.errorMessage}>{errorMessage}</Text>
        
        <TouchableOpacity style={styles.retryButton} onPress={handleDone} activeOpacity={0.8}>
          <Text style={styles.retryButtonText}>Try Again</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (jobStatus === 'completed') {
    return (
      <View style={[styles.container, { paddingBottom: insets.bottom }]}>
        {/* Header */}
        <View style={styles.successHeader}>
          <View style={styles.successIcon}>
            <Ionicons name="checkmark" size={32} color={colors.text} />
          </View>
          <Text style={styles.successTitle}>Panorama Ready!</Text>
          <Text style={styles.successCount}>
            {panoramas.length} result{panoramas.length !== 1 ? 's' : ''} generated
          </Text>
        </View>

        {/* Results List */}
        <FlatList
          data={panoramas}
          keyExtractor={(item) => item.id.toString()}
          contentContainerStyle={styles.resultsList}
          renderItem={({ item }) => (
            <TouchableOpacity 
              style={styles.resultCard} 
              onPress={() => handlePreview(item.filename)}
              activeOpacity={0.9}
            >
              <Image 
                source={{ uri: `${apiBaseUrl}/stitch/preview/${currentSessionId}/${item.filename}` }} 
                style={styles.resultImage}
                resizeMode="cover"
              />
              <View style={styles.resultOverlay}>
                <Text style={styles.resultDimensions}>
                  {item.shape[1]} × {item.shape[0]}
                </Text>
                <View style={styles.previewBadge}>
                  <Text style={styles.previewBadgeText}>Tap to preview</Text>
                  <Ionicons name="expand-outline" size={14} color={colors.text} />
                </View>
              </View>
            </TouchableOpacity>
          )}
        />

        {/* Done Button */}
        <View style={styles.footer}>
          <TouchableOpacity style={styles.doneButton} onPress={handleDone} activeOpacity={0.8}>
            <Text style={styles.doneButtonText}>Start New</Text>
            <Ionicons name="camera-outline" size={20} color={colors.text} />
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.centerContainer}>
      <Text style={styles.idleText}>No active job</Text>
      <TouchableOpacity style={styles.retryButton} onPress={handleDone} activeOpacity={0.8}>
        <Text style={styles.retryButtonText}>Go Home</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  centerContainer: {
    flex: 1,
    backgroundColor: colors.background,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.xl,
  },
  loaderContainer: {
    width: 80,
    height: 80,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.lg,
  },
  statusTitle: {
    ...typography.h2,
    color: colors.text,
    marginBottom: spacing.xs,
  },
  statusDescription: {
    ...typography.body,
    color: colors.textSecondary,
  },
  errorIcon: {
    marginBottom: spacing.lg,
  },
  errorTitle: {
    ...typography.h2,
    color: colors.error,
    marginBottom: spacing.sm,
  },
  errorMessage: {
    ...typography.body,
    color: colors.textSecondary,
    textAlign: 'center',
    marginBottom: spacing.xl,
  },
  retryButton: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    backgroundColor: colors.surface,
    borderRadius: borderRadius.full,
  },
  retryButtonText: {
    ...typography.bodyBold,
    color: colors.text,
  },
  idleText: {
    ...typography.body,
    color: colors.textSecondary,
    marginBottom: spacing.lg,
  },
  successHeader: {
    alignItems: 'center',
    paddingVertical: spacing.xl,
    paddingHorizontal: spacing.md,
  },
  successIcon: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: colors.success,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  successTitle: {
    ...typography.h2,
    color: colors.text,
    marginBottom: spacing.xs,
  },
  successCount: {
    ...typography.body,
    color: colors.textSecondary,
  },
  resultsList: {
    paddingHorizontal: spacing.md,
  },
  resultCard: {
    height: 200,
    borderRadius: borderRadius.lg,
    overflow: 'hidden',
    marginBottom: spacing.md,
    backgroundColor: colors.surface,
  },
  resultImage: {
    width: '100%',
    height: '100%',
  },
  resultOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: spacing.md,
    backgroundColor: colors.overlay,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  resultDimensions: {
    ...typography.caption,
    color: colors.textSecondary,
  },
  previewBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  previewBadgeText: {
    ...typography.small,
    color: colors.text,
  },
  footer: {
    padding: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.surface,
  },
  doneButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.md,
    borderRadius: borderRadius.lg,
    backgroundColor: colors.primary,
    gap: spacing.sm,
  },
  doneButtonText: {
    ...typography.bodyBold,
    color: colors.text,
  },
});
