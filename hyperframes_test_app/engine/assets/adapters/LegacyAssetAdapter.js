const path = require('path');
const AssetRequest = require('../models/AssetRequest');
const AssetResolutionService = require('../services/AssetResolutionService');

/**
 * LegacyAssetAdapter.js
 * Adapter layer converting legacy filename path strings into AssetRequests
 * and resolving them using the AssetResolutionService.
 */
class LegacyAssetAdapter {
  /**
   * Resolves a raw filename string into a standardized Asset object.
   * @param {string} filename 
   * @param {object} context 
   * @returns {Asset}
   */
  static resolveFilename(filename, context = {}) {
    if (!filename) return null;

    // Parse extension and base filename
    const ext = path.extname(filename).replace('.', '').toLowerCase();
    const base = path.basename(filename, path.extname(filename));

    // Guess category from extension
    let category = 'image';
    if (['wav', 'mp3', 'ogg', 'm4a'].includes(ext)) {
      category = 'audio';
    } else if (['svg'].includes(ext)) {
      category = 'diagram';
    } else if (['mp4', 'webm', 'mov', 'avi'].includes(ext)) {
      category = 'video';
    }

    // Build the request object
    const request = new AssetRequest({
      concept: base,
      subject: context.subject || 'general',
      category: category,
      preferred_format: ext,
      metadata: context.metadata || {}
    });

    // Run request through Resolution Service
    const resolvedAsset = AssetResolutionService.resolve(request);

    // If it's a fallback asset structure, preserve the original filename path
    // to maintain 100% backward compatibility for file references
    if (resolvedAsset.status === 'fallback') {
      resolvedAsset.path = filename;
    }

    return resolvedAsset;
  }
}

module.exports = LegacyAssetAdapter;
