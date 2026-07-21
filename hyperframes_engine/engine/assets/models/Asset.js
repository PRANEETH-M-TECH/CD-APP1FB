/**
 * Asset.js
 * Standardized resolved asset data model returned by the Asset Resolution Service.
 */
class Asset {
  constructor(fields = {}) {
    this.id = fields.id || `asset_${Math.random().toString(36).substr(2, 9)}`;
    this.provider = fields.provider || 'local';
    this.path = fields.path || '';
    this.format = fields.format || '';
    this.dimensions = fields.dimensions || { width: 0, height: 0 };
    this.metadata = fields.metadata || {};
    this.quality = fields.quality || 'medium';
    this.source = fields.source || 'local';
    this.license = fields.license || 'proprietary';
    this.status = fields.status || 'resolved';
    this.futureCacheInformation = fields.future_cache_info || {};
  }

  /**
   * Serializes the Asset into a pure JSON object.
   * @returns {object}
   */
  serialize() {
    return {
      id: this.id,
      provider: this.provider,
      path: this.path,
      format: this.format,
      dimensions: this.dimensions,
      metadata: this.metadata,
      quality: this.quality,
      source: this.source,
      license: this.license,
      status: this.status,
      future_cache_info: this.futureCacheInformation
    };
  }

  /**
   * Deserializes a JSON object into an Asset model.
   * @param {object} json 
   * @returns {Asset}
   */
  static deserialize(json) {
    if (!json) return null;
    return new Asset({
      id: json.id,
      provider: json.provider,
      path: json.path,
      format: json.format,
      dimensions: json.dimensions,
      metadata: json.metadata,
      quality: json.quality,
      source: json.source,
      license: json.license,
      status: json.status,
      future_cache_info: json.future_cache_info
    });
  }
}

module.exports = Asset;
