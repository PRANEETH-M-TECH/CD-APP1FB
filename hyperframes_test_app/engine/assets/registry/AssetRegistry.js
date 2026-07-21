const registeredAssets = [];

/**
 * AssetRegistry.js
 * Central library repository holding available asset metadata definitions.
 * Supports registration, indexing, and candidate queries.
 */
class AssetRegistry {
  /**
   * Registers a new asset metadata definition into the central library.
   * @param {object} assetData 
   */
  static register(assetData) {
    if (!assetData || !assetData.id || !assetData.concept) {
      throw new Error("Asset registration failed: Missing required fields (id, concept).");
    }
    registeredAssets.push({
      id: assetData.id,
      concept: assetData.concept.toLowerCase(),
      subject: (assetData.subject || 'general').toLowerCase(),
      category: (assetData.category || 'image').toLowerCase(),
      format: (assetData.format || '').toLowerCase(),
      path: assetData.path || '',
      dimensions: assetData.dimensions || { width: 0, height: 0 },
      metadata: assetData.metadata || {},
      quality: assetData.quality || 'medium',
      tags: (assetData.tags || []).map(t => t.toLowerCase()),
      license: assetData.license || 'proprietary'
    });
  }

  /**
   * Retrieves all registered asset metadata records.
   * @returns {Array<object>}
   */
  static getAll() {
    return registeredAssets;
  }

  /**
   * Query records matching candidate requirements.
   * @param {string} concept 
   * @param {string} subject 
   * @returns {Array<object>}
   */
  static query(concept, subject) {
    const cleanConcept = (concept || '').toLowerCase().trim();
    const cleanSubject = (subject || '').toLowerCase().trim();

    return registeredAssets.filter((asset) => {
      // 1. Concept matches main ID, concept string, or tags list
      const matchesConcept = asset.id.toLowerCase() === cleanConcept ||
                            asset.concept === cleanConcept ||
                            asset.tags.includes(cleanConcept) ||
                            cleanConcept.includes(asset.concept);

      // 2. Optional subject scoping
      const matchesSubject = !cleanSubject ||
                            cleanSubject === 'general' ||
                            asset.subject === 'general' ||
                            asset.subject === cleanSubject;

      return matchesConcept && matchesSubject;
    });
  }
}

module.exports = AssetRegistry;

// Pre-populate Registry with local project asset templates and audio references
AssetRegistry.register({
  id: 'stomach_diagram',
  concept: 'stomach',
  subject: 'biology',
  category: 'diagram',
  format: 'svg',
  path: './shared/stomach.svg',
  tags: ['organ', 'digestion', 'digestive_system']
});

AssetRegistry.register({
  id: 'intestines_diagram',
  concept: 'intestines',
  subject: 'biology',
  category: 'diagram',
  format: 'svg',
  path: './shared/intestines.svg',
  tags: ['organ', 'large_intestine', 'small_intestine', 'digestion']
});

// Pre-register dynamic audio tracks referencing scene narrations
for (let i = 1; i <= 24; i++) {
  AssetRegistry.register({
    id: `scene_audio_${i}`,
    concept: `scene_${i}`,
    subject: 'general',
    category: 'audio',
    format: 'wav',
    path: `./scene_${i}.wav`,
    tags: [`scene_audio_${i}`, 'tts', 'narration']
  });
}
