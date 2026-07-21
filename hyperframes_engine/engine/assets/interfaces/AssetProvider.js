/**
 * AssetProvider.js
 * Base interface class representing an asset data source provider.
 * Providers can resolve semantic concept searches against their libraries.
 */
class AssetProvider {
  constructor(name = 'BaseProvider') {
    this.name = name;
  }

  /**
   * Searches candidates for a given AssetRequest.
   * @param {AssetRequest} request 
   * @returns {Array<Asset>}
   */
  search(request) {
    throw new Error('Method search() must be implemented.');
  }

  /**
   * Resolves the best candidate matching the AssetRequest.
   * @param {AssetRequest} request 
   * @returns {Asset|null}
   */
  resolve(request) {
    throw new Error('Method resolve() must be implemented.');
  }

  /**
   * Verifies if an asset ID exists inside the provider database.
   * @param {string} assetId 
   * @returns {boolean}
   */
  exists(assetId) {
    throw new Error('Method exists() must be implemented.');
  }

  /**
   * Loads/Reads the asset payload.
   * @param {string} assetId 
   * @returns {any}
   */
  load(assetId) {
    throw new Error('Method load() must be implemented.');
  }

  /**
   * Retrieves specific provider metadata for an asset.
   * @param {string} assetId 
   * @returns {object}
   */
  metadata(assetId) {
    throw new Error('Method metadata() must be implemented.');
  }
}

module.exports = AssetProvider;
