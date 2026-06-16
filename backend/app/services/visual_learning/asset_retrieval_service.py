import logging
import httpx
import urllib.parse

logger = logging.getLogger(__name__)

USER_AGENT = "ChaduvuGuruVisualLearning/1.0 (contact@chaduvuguru.com)"

def is_valid_image_url(url: str) -> bool:
    """
    Verify if the URL path ends with a standard image format extension.
    """
    if not url:
        return False
    # Strip any query parameters or fragments
    path = url.split("?")[0].split("#")[0].lower()
    valid_extensions = (".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif", ".bmp")
    return path.endswith(valid_extensions)

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
        "gsrlimit": 5,  # Fetch slightly more to filter non-images
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
                        if img_url and is_valid_image_url(img_url):
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
                        if img_url and is_valid_image_url(img_url):
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
        "page_size": 5  # Fetch slightly more to filter non-images
    }
    headers = {"User-Agent": USER_AGENT}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                for result in results:
                    img_url = result.get("url")
                    if img_url and is_valid_image_url(img_url):
                        logger.info(f"[AssetRetrieval] Openverse found asset for '{query}': {img_url}")
                        return img_url
    except Exception as e:
        logger.error(f"[AssetRetrieval] Openverse search exception for '{query}': {e}", exc_info=True)
        
    return None

THEME_COLORS = {
    "indigo": "#6366f1",
    "gold": "#f59e0b",
    "emerald": "#10b981",
    "rose": "#ef4444"
}

async def get_lucide_icon_svg(icon_name: str, color_hex: str) -> str:
    """
    Fetch raw SVG from Lucide GitHub repository, inject the theme color,
    and return it as a data URL.
    """
    url = f"https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/{icon_name}.svg"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url, headers={"User-Agent": USER_AGENT})
            if response.status_code == 200:
                svg_content = response.text
                # Replace currentColor with theme color for strokes and fills
                svg_colored = svg_content.replace('stroke="currentColor"', f'stroke="{color_hex}"')
                svg_colored = svg_colored.replace('fill="currentColor"', f'fill="{color_hex}"')
                # URL encode the SVG content for a safe data: URI
                encoded_svg = urllib.parse.quote(svg_colored)
                return f"data:image/svg+xml;utf8,{encoded_svg}"
            else:
                logger.warning(f"[IconRetrieval] Lucide icon '{icon_name}' not found (Status: {response.status_code})")
    except Exception as e:
        logger.error(f"[IconRetrieval] Exception fetching Lucide icon '{icon_name}': {e}", exc_info=True)
    return None

async def retrieve_asset_url(query: str, asset_type: str = "image", theme: str = "indigo") -> str:
    """
    Orchestrate asset retrieval by querying Wikimedia, Openverse, or the Lucide Icon library.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return "/static/favicon.svg"
        
    theme_lower = theme.lower().strip()
    color_hex = THEME_COLORS.get(theme_lower, "#6366f1")

    # 1. Resolve Icons
    if asset_type == "icon":
        # Clean the icon query to match typical lucide names
        icon_name = cleaned_query.lower().replace("_", "-").replace(" ", "-")
        if icon_name.startswith("icon-"):
            icon_name = icon_name[5:]
        elif icon_name.startswith("icon:"):
            icon_name = icon_name[5:]
            
        svg_data_url = await get_lucide_icon_svg(icon_name, color_hex)
        if svg_data_url:
            return svg_data_url
        else:
            # Fallback icon if Lucide fetch fails
            color_encoded = urllib.parse.quote(color_hex)
            return f"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='{color_encoded}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'></circle><line x1='12' y1='16' x2='12' y2='12'></line><line x1='12' y1='8' x2='12.01' y2='8'></line></svg>"

    # 2. Intercept simple divider / conveyor / flowchart lines
    query_lower = cleaned_query.lower()
    if "line" in query_lower or "divider" in query_lower or "connector" in query_lower:
        color_encoded = urllib.parse.quote(color_hex)
        if "vertical" in query_lower:
            # Return a vertical SVG line data URL colored by theme
            return f"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 100' preserveAspectRatio='none'><line x1='5' y1='0' x2='5' y2='100' stroke='{color_encoded}' stroke-width='4' stroke-linecap='round'/></svg>"
        else:
            # Return a horizontal SVG line data URL colored by theme
            return f"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 10' preserveAspectRatio='none'><line x1='0' y1='5' x2='100' y2='5' stroke='{color_encoded}' stroke-width='4' stroke-linecap='round'/></svg>"

    # 3. Query Wikimedia
    url = await search_wikimedia(cleaned_query)
    if url:
        return url
        
    # 4. Query Openverse
    url = await search_openverse(cleaned_query)
    if url:
        return url
        
    # Ultimate static fallback
    logger.warning(f"[AssetRetrieval] No online asset found for '{cleaned_query}'. Using local fallback.")
    return "/static/favicon.svg"

