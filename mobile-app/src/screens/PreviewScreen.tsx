import React, { useState, useEffect, useRef, useCallback } from 'react';
import { View, StyleSheet, ActivityIndicator, Dimensions, Image, TouchableOpacity, Text, ScrollView } from 'react-native';
import { Gyroscope } from 'expo-sensors';
import * as ScreenOrientation from 'expo-screen-orientation';
import Animated, { useSharedValue, useAnimatedStyle, runOnJS } from 'react-native-reanimated';
import { useRoute, useNavigation } from '@react-navigation/native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useAppStore } from '../store/useAppStore';
import { colors, spacing, borderRadius, typography } from '../utils/theme';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

export default function PreviewScreen() {
  const route = useRoute<any>();
  const navigation = useNavigation();
  const insets = useSafeAreaInsets();
  const { filename } = route.params;
  const { apiBaseUrl, currentSessionId } = useAppStore();
  
  const imageUrl = `${apiBaseUrl}/stitch/result/${currentSessionId}/${filename}`;
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [loading, setLoading] = useState(true);
  const [gyroEnabled, setGyroEnabled] = useState(true);
  const [maxScroll, setMaxScroll] = useState(0);
  const [screenDimensions, setScreenDimensions] = useState({ width: SCREEN_WIDTH, height: SCREEN_HEIGHT });

  // Animation value for horizontal position
  const positionX = useSharedValue(0);
  const subscription = useRef<any>(null);
  const scrollViewRef = useRef<ScrollView>(null);

  // Force landscape on mount, restore portrait on unmount
  useEffect(() => {
    const lockLandscape = async () => {
      await ScreenOrientation.lockAsync(ScreenOrientation.OrientationLock.LANDSCAPE);
      const { width, height } = Dimensions.get('window');
      setScreenDimensions({ width: Math.max(width, height), height: Math.min(width, height) });
    };
    
    lockLandscape();
    
    return () => {
      ScreenOrientation.lockAsync(ScreenOrientation.OrientationLock.PORTRAIT);
    };
  }, []);

  useEffect(() => {
    Image.getSize(imageUrl, (w, h) => {
      // Scale to fit screen height (landscape)
      const scaledHeight = screenDimensions.height;
      const scaledWidth = (w / h) * scaledHeight;
      setImageSize({ width: scaledWidth, height: scaledHeight });
      
      const scrollRange = Math.max(0, scaledWidth - screenDimensions.width);
      setMaxScroll(scrollRange);
      
      // Center initially
      positionX.value = -scrollRange / 2;
      setLoading(false);
    }, (err) => {
      console.error("Failed to load image", err);
      setLoading(false);
    });

    return () => _unsubscribe();
  }, [screenDimensions]);

  useEffect(() => {
    if (gyroEnabled && !loading && maxScroll > 0) {
      _subscribe();
    } else {
      _unsubscribe();
    }
    return () => _unsubscribe();
  }, [gyroEnabled, loading, maxScroll]);

  const _subscribe = () => {
    Gyroscope.setUpdateInterval(16);
    subscription.current = Gyroscope.addListener(({ y }) => {
      const sensitivity = 25;
      let newX = positionX.value - (y * sensitivity);
      newX = Math.min(0, Math.max(newX, -maxScroll));
      positionX.value = newX;
    });
  };

  const _unsubscribe = () => {
    if (subscription.current) {
      subscription.current.remove();
      subscription.current = null;
    }
  };

  const toggleGyro = () => {
    setGyroEnabled(!gyroEnabled);
  };

  const handleClose = async () => {
    await ScreenOrientation.lockAsync(ScreenOrientation.OrientationLock.PORTRAIT);
    navigation.goBack();
  };

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: positionX.value }],
  }));

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={colors.primary} />
        <Text style={styles.loadingText}>Loading panorama...</Text>
      </View>
    );
  }

  // When gyro is off, use ScrollView for touch panning
  if (!gyroEnabled) {
    return (
      <View style={styles.container}>
        <ScrollView
          ref={scrollViewRef}
          horizontal
          showsHorizontalScrollIndicator={false}
          contentOffset={{ x: maxScroll / 2, y: 0 }}
          style={styles.scrollView}
        >
          <Image 
            source={{ uri: imageUrl }}
            style={{ width: imageSize.width, height: imageSize.height }}
            resizeMode="cover"
          />
        </ScrollView>

        {/* Controls */}
        <View style={[styles.controlsTop, { paddingTop: insets.top + spacing.sm }]}>
          <TouchableOpacity style={styles.controlButton} onPress={handleClose} activeOpacity={0.8}>
            <Ionicons name="close" size={24} color={colors.text} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.controlButton} onPress={toggleGyro} activeOpacity={0.8}>
            <Ionicons name="hand-left-outline" size={24} color={colors.text} />
          </TouchableOpacity>
        </View>

        <View style={[styles.hint, { bottom: insets.bottom + spacing.lg }]}>
          <Ionicons name="swap-horizontal-outline" size={16} color={colors.textSecondary} />
          <Text style={styles.hintText}>Swipe to pan</Text>
        </View>
      </View>
    );
  }

  // Gyro mode with Animated.Image
  return (
    <View style={styles.container}>
      <Animated.Image 
        source={{ uri: imageUrl }}
        style={[styles.image, { width: imageSize.width, height: imageSize.height }, animatedStyle]}
        resizeMode="cover"
      />

      {/* Controls */}
      <View style={[styles.controlsTop, { paddingTop: insets.top + spacing.sm }]}>
        <TouchableOpacity style={styles.controlButton} onPress={handleClose} activeOpacity={0.8}>
          <Ionicons name="close" size={24} color={colors.text} />
        </TouchableOpacity>
        <TouchableOpacity style={[styles.controlButton, styles.controlButtonActive]} onPress={toggleGyro} activeOpacity={0.8}>
          <Ionicons name="compass" size={24} color={colors.text} />
        </TouchableOpacity>
      </View>

      <View style={[styles.hint, { bottom: insets.bottom + spacing.lg }]}>
        <Ionicons name="phone-landscape-outline" size={16} color={colors.textSecondary} />
        <Text style={styles.hintText}>Tilt device to pan</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  loadingContainer: {
    flex: 1,
    backgroundColor: colors.background,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    ...typography.body,
    color: colors.textSecondary,
    marginTop: spacing.md,
  },
  scrollView: {
    flex: 1,
  },
  image: {
    position: 'absolute',
    top: 0,
  },
  controlsTop: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.md,
    zIndex: 10,
  },
  controlButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: colors.overlay,
    justifyContent: 'center',
    alignItems: 'center',
  },
  controlButtonActive: {
    backgroundColor: colors.primary,
  },
  hint: {
    position: 'absolute',
    alignSelf: 'center',
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.overlay,
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.md,
    borderRadius: borderRadius.full,
    gap: spacing.xs,
  },
  hintText: {
    ...typography.small,
    color: colors.textSecondary,
  },
});
