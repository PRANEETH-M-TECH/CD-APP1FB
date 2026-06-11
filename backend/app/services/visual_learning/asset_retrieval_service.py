import logging
import httpx
import urllib.parse

logger = logging.getLogger(__name__)

USER_AGENT = "ChaduvuGuruVisualLearning/1.0 (contact@chaduvuguru.com)"

async def search_wikimedia(query: str) -> str:
    """
    Search Wikimedia Commons for an image representing the query.
    Returns the URL of the top 1 result, or None if not found.
    """
    url = "https://commons.wikimedia.org/w/api.php"
    headers = {"User-Agent": USER_AGENT}
    
    # Try search with bitmap/drawing restriction
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap|drawing {query}",
        "gsrnamespace": 6,
        "gsrlimit": 3,
        "prop": "imageinfo",
        "iiprop": "url",
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                pages = data.get("query", {}).get("pages", {})
                for page_id, page_data in pages.items():
                    imageinfo = page_data.get("imageinfo", [])
                    if imageinfo:
                        img_url = imageinfo[0].get("url")
                        if img_url:
                            logger.info(f"[AssetRetrieval] Wikimedia found asset for '{query}': {img_url}")
                            return img_url
            
            # Broad search without restrictions if restricted search failed
            params["gsrsearch"] = query
            response = await client.get(url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                pages = data.get("query", {}).get("pages", {})
                for page_id, page_data in pages.items():
                    imageinfo = page_data.get("imageinfo", [])
                    if imageinfo:
                        img_url = imageinfo[0].get("url")
                        if img_url:
                            logger.info(f"[AssetRetrieval] Wikimedia found asset (broad) for '{query}': {img_url}")
                            return img_url
                            
    except Exception as e:
        logger.error(f"[AssetRetrieval] Wikimedia search exception for '{query}': {e}", exc_info=True)
        
    return None

async def search_openverse(query: str) -> str:
    """
    Search WordPress Openverse API for the query.
    Returns the URL of the top 1 result, or None if not found.
    """
    url = "https://api.openverse.org/v1/images/"
    params = {
        "q": query,
        "page_size": 1
    }
    headers = {"User-Agent": USER_AGENT}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if results:
                    img_url = results[0].get("url")
                    if img_url:
                        logger.info(f"[AssetRetrieval] Openverse found asset for '{query}': {img_url}")
                        return img_url
    except Exception as e:
        logger.error(f"[AssetRetrieval] Openverse search exception for '{query}': {e}", exc_info=True)
        
    return None

async def retrieve_asset_url(query: str) -> str:
    """
    Orchestrate asset retrieval by querying Wikimedia first,
    falling back to Openverse, and ultimately using a default placeholder image if both fail.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return "/static/favicon.svg"
        
    # Query Wikimedia
    url = await search_wikimedia(cleaned_query)
    if url:
        return url
        
    # Query Openverse
    url = await search_openverse(cleaned_query)
    if url:
        return url
        
    # Ultimate static fallback
    logger.warning(f"[AssetRetrieval] No online asset found for '{cleaned_query}'. Using local fallback.")
    return "/static/favicon.svg"
