import React, { useState } from 'react';
import { View, Text, TextInput, Switch, StyleSheet, ScrollView, TouchableOpacity, Alert } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useAppStore } from '../store/useAppStore';
import { colors, spacing, borderRadius, typography } from '../utils/theme';

export default function SettingsScreen() {
  const insets = useSafeAreaInsets();
  const { 
    apiBaseUrl, setApiBaseUrl, 
    settings, updateSettings,
    checkConnection,
    isConnected
  } = useAppStore();
  
  const [urlInput, setUrlInput] = useState(apiBaseUrl);
  const [isTesting, setIsTesting] = useState(false);

  const handleTestConnection = async () => {
    setApiBaseUrl(urlInput);
    setIsTesting(true);
    const connected = await checkConnection();
    setIsTesting(false);
    
    if (connected) {
      Alert.alert("✓ Connected", "Successfully connected to PanoStitch server.");
    } else {
      Alert.alert("✕ Connection Failed", "Could not reach the server. Check the URL and ensure the server is running.");
    }
  };

  const SettingRow = ({ 
    label, 
    description, 
    value, 
    onValueChange 
  }: { 
    label: string; 
    description?: string; 
    value: boolean; 
    onValueChange: (val: boolean) => void;
  }) => (
    <View style={styles.settingRow}>
      <View style={styles.settingInfo}>
        <Text style={styles.settingLabel}>{label}</Text>
        {description && <Text style={styles.settingDescription}>{description}</Text>}
      </View>
      <Switch 
        value={value} 
        onValueChange={onValueChange}
        trackColor={{ false: colors.surfaceLight, true: colors.primaryLight }}
        thumbColor={value ? colors.primary : colors.textMuted}
      />
    </View>
  );

  return (
    <ScrollView 
      style={styles.container} 
      contentContainerStyle={{ paddingBottom: insets.bottom + spacing.xl }}
    >
      {/* Connection Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Server Connection</Text>
        
        <View style={styles.card}>
          <View style={styles.inputRow}>
            <View style={[styles.connectionDot, { backgroundColor: isConnected ? colors.success : colors.error }]} />
            <TextInput 
              style={styles.urlInput} 
              value={urlInput}
              onChangeText={setUrlInput}
              placeholder="http://192.168.1.x:5000/api"
              placeholderTextColor={colors.textMuted}
              autoCapitalize="none"
              autoCorrect={false}
            />
          </View>
          
          <TouchableOpacity 
            style={[styles.testButton, isTesting && styles.testButtonDisabled]} 
            onPress={handleTestConnection}
            disabled={isTesting}
            activeOpacity={0.8}
          >
            <Ionicons 
              name={isTesting ? "hourglass-outline" : "flash-outline"} 
              size={18} 
              color={colors.text} 
            />
            <Text style={styles.testButtonText}>
              {isTesting ? 'Testing...' : 'Test Connection'}
            </Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Stitching Options */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Stitching Algorithm</Text>
        
        <View style={styles.card}>
          <SettingRow
            label="Deep Learning Matcher"
            description="Uses DISK + LightGlue for better accuracy"
            value={settings.use_dnn}
            onValueChange={(val) => updateSettings({ use_dnn: val })}
          />
          
          <View style={styles.divider} />
          
          <SettingRow
            label="Harris Corners"
            description="Alternative corner detection method"
            value={settings.use_harris}
            onValueChange={(val) => updateSettings({ use_harris: val })}
          />
          
          <View style={styles.divider} />
          
          <SettingRow
            label="Cylindrical Projection"
            description="Better for wide panoramas"
            value={settings.use_cylindrical}
            onValueChange={(val) => updateSettings({ use_cylindrical: val })}
          />
        </View>
      </View>

      {/* Advanced Parameters */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Advanced</Text>
        
        <View style={styles.card}>
          <View style={styles.paramRow}>
            <View style={styles.paramInfo}>
              <Text style={styles.settingLabel}>Focal Length</Text>
              <Text style={styles.settingDescription}>Camera focal length in pixels</Text>
            </View>
            <TextInput 
              style={styles.paramInput} 
              value={String(settings.focal_length)}
              onChangeText={(val) => updateSettings({ focal_length: parseFloat(val) || 0 })}
              keyboardType="numeric"
              placeholderTextColor={colors.textMuted}
            />
          </View>
          
          <View style={styles.divider} />
          
          <View style={styles.paramRow}>
            <View style={styles.paramInfo}>
              <Text style={styles.settingLabel}>Max Dimension</Text>
              <Text style={styles.settingDescription}>Resize images before processing</Text>
            </View>
            <TextInput 
              style={styles.paramInput} 
              value={String(settings.resize)}
              onChangeText={(val) => updateSettings({ resize: parseInt(val) || 0 })}
              keyboardType="numeric"
              placeholderTextColor={colors.textMuted}
            />
          </View>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  section: {
    marginTop: spacing.lg,
    paddingHorizontal: spacing.md,
  },
  sectionTitle: {
    ...typography.caption,
    color: colors.textSecondary,
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: spacing.sm,
    marginLeft: spacing.xs,
  },
  card: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.lg,
    overflow: 'hidden',
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.surfaceLight,
  },
  connectionDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: spacing.sm,
  },
  urlInput: {
    flex: 1,
    ...typography.body,
    color: colors.text,
  },
  testButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.md,
    backgroundColor: colors.primary,
    gap: spacing.xs,
  },
  testButtonDisabled: {
    backgroundColor: colors.surfaceLight,
  },
  testButtonText: {
    ...typography.bodyBold,
    color: colors.text,
  },
  settingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: spacing.md,
  },
  settingInfo: {
    flex: 1,
    marginRight: spacing.md,
  },
  settingLabel: {
    ...typography.body,
    color: colors.text,
  },
  settingDescription: {
    ...typography.small,
    color: colors.textMuted,
    marginTop: 2,
  },
  divider: {
    height: 1,
    backgroundColor: colors.surfaceLight,
    marginLeft: spacing.md,
  },
  paramRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: spacing.md,
  },
  paramInfo: {
    flex: 1,
  },
  paramInput: {
    ...typography.body,
    color: colors.text,
    backgroundColor: colors.surfaceLight,
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.sm,
    borderRadius: borderRadius.sm,
    minWidth: 80,
    textAlign: 'center',
  },
});
