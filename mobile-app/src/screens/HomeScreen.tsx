import React, { useState, useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Image, Dimensions, Alert } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useNavigation } from '@react-navigation/native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useAppStore } from '../store/useAppStore';
import { colors, spacing, borderRadius, typography } from '../utils/theme';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export default function HomeScreen() {
  const navigation = useNavigation<any>();
  const insets = useSafeAreaInsets();
  const cameraRef = useRef<CameraView>(null);
  const [permission] = useCameraPermissions();
  
  const [capturedImages, setCapturedImages] = useState<string[]>([]);
  
  const { isConnected, checkConnection, startStitching } = useAppStore();

  React.useEffect(() => {
    checkConnection();
  }, []);

  const takePicture = async () => {
    if (cameraRef.current) {
      try {
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.8,
          skipProcessing: true
        });
        if (photo) {
          setCapturedImages([...capturedImages, photo.uri]);
        }
      } catch (e) {
        console.error("Failed to take picture", e);
      }
    }
  };

  const removeImage = (index: number) => {
    const newImages = [...capturedImages];
    newImages.splice(index, 1);
    setCapturedImages(newImages);
  };

  const handleStitch = async () => {
    if (capturedImages.length < 2) {
      Alert.alert("Not enough photos", "Please capture at least 2 photos.");
      return;
    }
    
    const connected = await checkConnection();
    if (!connected) {
      Alert.alert("Connection Error", "Cannot connect to server. Check Settings.");
      return;
    }
    
    startStitching(capturedImages);
    navigation.navigate('Status');
  };

  if (!permission || !permission.granted) {
    return (
      <View style={[styles.container, styles.centered]}>
        <Text style={styles.errorText}>Camera permission required</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Camera Viewfinder */}
      <CameraView style={styles.camera} ref={cameraRef} />

      {/* Top Bar */}
      <View style={[styles.topBar, { paddingTop: insets.top + spacing.sm }]}>
        {/* Connection Status */}
        <View style={styles.statusBadge}>
          <View style={[styles.statusDot, { backgroundColor: isConnected ? colors.success : colors.error }]} />
          <Text style={styles.statusText}>{isConnected ? 'Connected' : 'Offline'}</Text>
        </View>
        
        {/* Settings Button */}
        <TouchableOpacity 
          style={styles.iconButton} 
          onPress={() => navigation.navigate('Settings')}
          activeOpacity={0.7}
        >
          <Ionicons name="settings-outline" size={24} color={colors.text} />
        </TouchableOpacity>
      </View>

      {/* Bottom Controls */}
      <View style={[styles.bottomControls, { paddingBottom: insets.bottom + spacing.md }]}>
        {/* Image Gallery */}
        {capturedImages.length > 0 && (
          <View style={styles.gallerySection}>
            <Text style={styles.galleryLabel}>
              {capturedImages.length} photo{capturedImages.length !== 1 ? 's' : ''} captured
            </Text>
            <ScrollView 
              horizontal 
              showsHorizontalScrollIndicator={false} 
              contentContainerStyle={styles.galleryScroll}
            >
              {capturedImages.map((uri, index) => (
                <View key={index} style={styles.thumbnailContainer}>
                  <Image source={{ uri }} style={styles.thumbnail} />
                  <TouchableOpacity 
                    style={styles.removeButton} 
                    onPress={() => removeImage(index)}
                    activeOpacity={0.8}
                  >
                    <Ionicons name="close" size={14} color={colors.text} />
                  </TouchableOpacity>
                  <View style={styles.thumbnailIndex}>
                    <Text style={styles.thumbnailIndexText}>{index + 1}</Text>
                  </View>
                </View>
              ))}
            </ScrollView>
          </View>
        )}

        {/* Action Buttons */}
        <View style={styles.actionRow}>
          {/* Stitch Button (left) */}
          <View style={styles.sideAction}>
            {capturedImages.length >= 2 && (
              <TouchableOpacity style={styles.stitchButton} onPress={handleStitch} activeOpacity={0.8}>
                <Ionicons name="git-merge" size={20} color={colors.background} />
                <Text style={styles.stitchButtonText}>Stitch</Text>
              </TouchableOpacity>
            )}
          </View>

          {/* Capture Button (center) */}
          <TouchableOpacity 
            style={styles.captureButton} 
            onPress={takePicture}
            activeOpacity={0.9}
          >
            <View style={styles.captureButtonOuter}>
              <View style={styles.captureButtonInner} />
            </View>
          </TouchableOpacity>

          {/* Counter (right) */}
          <View style={styles.sideAction}>
            {capturedImages.length > 0 && (
              <View style={styles.counterBadge}>
                <Text style={styles.counterText}>{capturedImages.length}</Text>
              </View>
            )}
          </View>
        </View>

        {/* Helper Text */}
        <Text style={styles.helperText}>
          {capturedImages.length === 0 
            ? 'Take overlapping photos for best results'
            : capturedImages.length === 1 
              ? 'Take at least one more photo'
              : 'Ready to stitch!'
          }
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  centered: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  errorText: {
    ...typography.body,
    color: colors.textSecondary,
  },
  camera: {
    ...StyleSheet.absoluteFillObject,
  },
  topBar: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    zIndex: 10,
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.overlay,
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.sm,
    borderRadius: borderRadius.full,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: spacing.xs,
  },
  statusText: {
    ...typography.small,
    color: colors.text,
  },
  iconButton: {
    width: 44,
    height: 44,
    borderRadius: borderRadius.full,
    backgroundColor: colors.overlay,
    justifyContent: 'center',
    alignItems: 'center',
  },
  bottomControls: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: colors.overlay,
    paddingTop: spacing.lg,
    zIndex: 10,
  },
  gallerySection: {
    marginBottom: spacing.md,
  },
  galleryLabel: {
    ...typography.caption,
    color: colors.textSecondary,
    marginLeft: spacing.md,
    marginBottom: spacing.sm,
  },
  galleryScroll: {
    paddingHorizontal: spacing.md,
  },
  thumbnailContainer: {
    marginRight: spacing.sm,
    position: 'relative',
  },
  thumbnail: {
    width: 64,
    height: 80,
    borderRadius: borderRadius.sm,
    borderWidth: 2,
    borderColor: colors.surfaceLight,
  },
  removeButton: {
    position: 'absolute',
    top: -6,
    right: -6,
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: colors.error,
    justifyContent: 'center',
    alignItems: 'center',
  },
  thumbnailIndex: {
    position: 'absolute',
    bottom: 4,
    left: 4,
    backgroundColor: colors.overlay,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: borderRadius.sm,
  },
  thumbnailIndexText: {
    ...typography.small,
    color: colors.text,
  },
  actionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.xl,
    marginBottom: spacing.md,
  },
  sideAction: {
    width: 100,
    alignItems: 'center',
  },
  stitchButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    borderRadius: borderRadius.full,
    backgroundColor: colors.accent,
    gap: spacing.xs,
  },
  stitchButtonText: {
    ...typography.bodyBold,
    color: colors.background,
  },
  captureButton: {
    width: 80,
    height: 80,
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonOuter: {
    width: 72,
    height: 72,
    borderRadius: 36,
    borderWidth: 4,
    borderColor: colors.text,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'transparent',
  },
  captureButtonInner: {
    width: 58,
    height: 58,
    borderRadius: 29,
    backgroundColor: colors.text,
  },
  counterBadge: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.surface,
    justifyContent: 'center',
    alignItems: 'center',
  },
  counterText: {
    ...typography.bodyBold,
    color: colors.text,
  },
  helperText: {
    ...typography.caption,
    color: colors.textMuted,
    textAlign: 'center',
  },
});
