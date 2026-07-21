/**
 * AssetRequest.js
 * Extensible model representing an asset search request.
 * Contains semantic concepts, categorization, formats, and design constraints.
 */
class AssetRequest {
  constructor(fields = {}) {
    this.concept = fields.concept || '';
    this.subject = fields.subject || 'general';
    this.category = fields.category || 'image'; // image, audio, video, diagram
    this.style = fields.style || 'standard';
    this.preferredFormat = fields.preferred_format || '';
    this.quality = fields.quality || 'medium';
    this.theme = fields.theme || 'indigo';
    this.language = fields.language || 'en';
    this.tags = fields.tags || [];
    this.metadata = fields.metadata || {};
    this.futureConstraints = fields.future_constraints || {};
  }
}

module.exports = AssetRequest;
