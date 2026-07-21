const AssetProvider = require('../interfaces/AssetProvider');
const AssetRegistry = require('../registry/AssetRegistry');
const Asset = require('../models/Asset');

/**
 * LocalPNGProvider.js
 * Extends AssetProvider to resolve local raster files (PNG, JPG) and audio clips (WAV).
 */
class LocalPNGProvider extends AssetProvider {
  constructor() {
    super('LocalPNGProvider');
  }

  /**
   * Searches for candidate assets matching request.
   * @param {AssetRequest} request 
   * @returns {Array<Asset>}
   */
  search(request) {
    // Skip if search prefers SVG explicitly
    if (request.preferredFormat && request.preferredFormat.toLowerCase() === 'svg') {
      return [];
    }

    const matches = AssetRegistry.query(request.concept, request.subject);
    return matches
      .filter(c => c.format !== 'svg')
      .map(c => new Asset({
        id: c.id,
        provider: this.name,
        path: c.path,
        format: c.format,
        dimensions: c.dimensions,
        metadata: c.metadata,
        quality: c.quality,
        source: 'local',
        license: c.license,
        status: 'resolved'
      }));
  }

  resolve(request) {
    const candidates = this.search(request);
    return candidates.length > 0 ? candidates[0] : null;
  }

  exists(assetId) {
    return AssetRegistry.getAll().some(c => c.id === assetId && c.format !== 'svg');
  }

  load(assetId) {
    const assetRecord = AssetRegistry.getAll().find(c => c.id === assetId && c.format !== 'svg');
    if (!assetRecord) return null;
    const fs = require('fs');
    const path = require('path');
    const fullPath = path.resolve(assetRecord.path);
    return fs.existsSync(fullPath) ? fs.readFileSync(fullPath) : null;
  }

  metadata(assetId) {
    const record = AssetRegistry.getAll().find(c => c.id === assetId && c.format !== 'svg');
    return record ? record.metadata : {};
  }
}

module.exports = LocalPNGProvider;
