const LocalSVGProvider = require('../providers/LocalSVGProvider');
const LocalPNGProvider = require('../providers/LocalPNGProvider');
const Asset = require('../models/Asset');

const providers = [];

/**
 * AssetResolutionService.js
 * Central resolution service querying and ranking candidates from registered providers
 * to return the best matched standardized Asset object for an AssetRequest.
 */
class AssetResolutionService {
  /**
   * Registers an AssetProvider instance.
   * @param {AssetProvider} provider 
   */
  static registerProvider(provider) {
    providers.push(provider);
  }

  /**
   * Returns all currently registered provider instances.
   * @returns {Array<AssetProvider>}
   */
  static getProviders() {
    return providers;
  }

  /**
   * Resolves the best matched Asset candidate for a given request.
   * @param {AssetRequest} request 
   * @returns {Asset}
   */
  static resolve(request) {
    let allCandidates = [];

    // Query all registered providers
    for (const provider of providers) {
      try {
        const candidates = provider.search(request);
        if (candidates && candidates.length > 0) {
          allCandidates = allCandidates.concat(candidates);
        }
      } catch (err) {
        console.error(`[AssetResolutionService Error] Querying provider ${provider.name} failed:`, err);
      }
    }

    // Fallback default resolved asset if no provider matched
    if (allCandidates.length === 0) {
      const format = request.preferredFormat || (request.category === 'audio' ? 'wav' : 'png');
      const conceptName = request.concept || 'default_asset';
      
      // Determine probable local path
      let path = '';
      if (request.category === 'audio') {
        path = `./${conceptName}.${format}`;
      } else {
        path = `./shared/${conceptName}.${format}`;
      }

      return new Asset({
        id: `fallback_${conceptName}`,
        provider: 'fallback',
        path: path,
        format: format,
        metadata: { concept: conceptName, subject: request.subject },
        status: 'fallback'
      });
    }

    // Rank candidates by matching scores
    allCandidates.sort((a, b) => {
      let scoreA = 0;
      let scoreB = 0;

      // 1. Preferred format matching
      if (request.preferredFormat) {
        const reqFormat = request.preferredFormat.toLowerCase();
        if (a.format.toLowerCase() === reqFormat) scoreA += 10;
        if (b.format.toLowerCase() === reqFormat) scoreB += 10;
      }

      // 2. Subject matching
      if (request.subject) {
        const reqSubject = request.subject.toLowerCase();
        const subA = (a.metadata.subject || '').toLowerCase();
        const subB = (b.metadata.subject || '').toLowerCase();
        if (subA === reqSubject) scoreA += 5;
        if (subB === reqSubject) scoreB += 5;
      }

      // 3. Quality preference matching
      if (request.quality) {
        if (a.quality === request.quality) scoreA += 2;
        if (b.quality === request.quality) scoreB += 2;
      }

      return scoreB - scoreA; // descending order
    });

    return allCandidates[0];
  }
}

// Automatically register local local providers on service load
AssetResolutionService.registerProvider(new LocalSVGProvider());
AssetResolutionService.registerProvider(new LocalPNGProvider());

module.exports = AssetResolutionService;
