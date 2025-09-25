# ============================================================================
# NAMASTE to ICD-11 TM2 Mapping API
# ============================================================================
# Flask API for mapping NAMASTE Ayurveda diagnostic codes to ICD-11 TM2 codes
# Provides RESTful endpoints for healthcare interoperability

# ============================================================================
# IMPORTS
# ============================================================================

# Third-party imports
from flask import Flask, request, jsonify, render_template

# Standard library imports
import time
import logging
import json
import uuid
from datetime import datetime
from collections import defaultdict

# ============================================================================
# AUDIT LOGGING SYSTEM
# ============================================================================

# In-memory audit log storage (use database in production)
audit_log = []
audit_stats = defaultdict(int)

def log_audit_event(operation, request_data=None, response_data=None, status_code=200, error=None):
    """
    Log an audit event for compliance and monitoring.
    
    Args:
        operation (str): The operation being performed
        request_data (dict): Request payload data
        response_data (dict): Response data
        status_code (int): HTTP status code
        error (str): Error message if applicable
    """
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    # Extract client information from Flask request context
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
    user_agent = request.headers.get('User-Agent', 'unknown')
    
    # Extract token info (hash for privacy)
    auth_header = request.headers.get('Authorization', '')
    token_hash = None
    if auth_header:
        import hashlib
        token = auth_header.replace('Bearer ', '')
        token_hash = 'sha256:' + hashlib.sha256(token.encode()).hexdigest()[:16]
    
    audit_entry = {
        'id': str(uuid.uuid4()),
        'timestamp': timestamp,
        'operation': operation,
        'client': {
            'ip': client_ip,
            'user_agent': user_agent,
            'token_hash': token_hash
        },
        'request': {
            'method': request.method,
            'url': request.url,
            'endpoint': request.endpoint,
            'data': request_data
        },
        'response': {
            'status_code': status_code,
            'data_size': len(json.dumps(response_data)) if response_data else 0,
            'has_error': error is not None
        },
        'error': error,
        'processing_time': getattr(request, '_audit_start_time', None)
    }
    
    # Calculate processing time if available
    if hasattr(request, '_audit_start_time'):
        audit_entry['processing_time'] = time.time() - request._audit_start_time
    
    # Store audit entry
    audit_log.insert(0, audit_entry)
    
    # Update statistics
    audit_stats['total_requests'] += 1
    audit_stats[f'status_{status_code}'] += 1
    audit_stats[f'operation_{operation}'] += 1
    if error:
        audit_stats['total_errors'] += 1
    
    # Keep only last 10000 entries in memory
    if len(audit_log) > 10000:
        audit_log.pop()
    
    logger.info(f"Audit: {operation} | Status: {status_code} | Client: {client_ip}")


# ============================================================================
# APPLICATION SETUP
# ============================================================================

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.before_request
def before_request_audit():
    """Set up audit tracking for the request."""
    request._audit_start_time = time.time()


@app.after_request
def after_request_audit(response):
    """Log audit event after request completion."""
    # Skip audit logging for static files and health checks
    if request.endpoint in ['static']:
        return response
    
    try:
        # Try to get response data for audit
        response_data = None
        if response.content_type == 'application/json':
            try:
                response_data = response.get_json()
            except:
                response_data = {'content_type': response.content_type}
        
        # Get request data
        request_data = None
        if request.is_json:
            request_data = request.get_json(silent=True)
        elif request.form:
            request_data = dict(request.form)
        elif request.args:
            request_data = dict(request.args)
        
        # Log the audit event
        error_msg = None
        if response.status_code >= 400:
            error_msg = f"HTTP {response.status_code}"
            if response_data and 'message' in response_data:
                error_msg = response_data['message']
        
        log_audit_event(
            operation=request.endpoint or 'unknown',
            request_data=request_data,
            response_data=response_data,
            status_code=response.status_code,
            error=error_msg
        )
    except Exception as e:
        logger.error(f"Failed to log audit event: {e}")
    
    return response

# ============================================================================
# INITIALIZATION
# ============================================================================

logger.info("Loading NAMASTE mapping resources...")
startup_start = time.time()


import main
# Warm up the mapping system with a test query
main.fetch_namaste_entry('ABB-10')
startup_time = time.time() - startup_start
logger.info(f"Startup complete in {startup_time:.2f} seconds")

# ============================================================================
# AUTHENTICATION & MIDDLEWARE
# ============================================================================

def check_token(token):
    """
    Validate authentication token.
    
    TODO: Integrate with ABHA OAuth 2.0 validation system
    
    Args:
        token (str): Bearer token from Authorization header
        
    Returns:
        bool: True if token is valid, False otherwise
    """
    # Placeholder implementation - replace with actual OAuth validation
    return token == "sparrow"


@app.before_request
def auth_check():
    """
    Pre-request authentication middleware.
    
    Skips authentication for health check endpoint and static files.
    Returns 401 for invalid tokens.
    """
    # Skip auth for health check, frontend, and static files
    if request.endpoint in ['health', 'frontend', 'static']:
        return
        
    # Extract and validate token
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "")
    
    if not check_token(token):
        return jsonify({"error": "unauthorized", "message": "Invalid or missing token"}), 401

# ============================================================================
# FRONTEND & API ENDPOINTS
# ============================================================================

@app.route("/", methods=["GET"])
def frontend():
    """
    Serve the frontend interface.
    
    Returns:
        HTML page with API testing interface
    """
    return render_template('index.html')

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    
    Returns:
        dict: Service status
    """
    return {"status": "ok", "service": "NAMASTE-ICD11 Mapping API"}


@app.route("/audit", methods=["GET"])
def get_audit_logs():
    """
    Get audit logs with filtering and pagination.
    
    Query Parameters:
        page (int): Page number (default: 1)
        limit (int): Items per page (default: 50, max: 1000)
        operation (str): Filter by operation name
        status (int): Filter by status code
        start_time (str): Filter events after this ISO timestamp
        end_time (str): Filter events before this ISO timestamp
        
    Returns:
        JSON response with audit logs and metadata
    """
    try:
        # Get query parameters
        page = max(1, int(request.args.get('page', 1)))
        limit = min(1000, max(1, int(request.args.get('limit', 50))))
        operation_filter = request.args.get('operation')
        status_filter = request.args.get('status')
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        
        # Filter audit logs
        filtered_logs = audit_log.copy()
        
        if operation_filter:
            filtered_logs = [log for log in filtered_logs if log['operation'] == operation_filter]
        
        if status_filter:
            status_code = int(status_filter)
            filtered_logs = [log for log in filtered_logs if log['response']['status_code'] == status_code]
        
        if start_time:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            filtered_logs = [log for log in filtered_logs 
                           if datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00')) >= start_dt]
        
        if end_time:
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            filtered_logs = [log for log in filtered_logs 
                           if datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00')) <= end_dt]
        
        # Paginate results
        total_count = len(filtered_logs)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_logs = filtered_logs[start_idx:end_idx]
        
        # Calculate pagination metadata
        total_pages = (total_count + limit - 1) // limit
        
        return jsonify({
            "status": "success",
            "data": {
                "logs": paginated_logs,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total_pages": total_pages,
                    "total_count": total_count,
                    "has_next": page < total_pages,
                    "has_prev": page > 1
                },
                "statistics": dict(audit_stats),
                "filters_applied": {
                    "operation": operation_filter,
                    "status": status_filter,
                    "start_time": start_time,
                    "end_time": end_time
                }
            }
        })
        
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": f"Invalid parameter format: {str(e)}"
        }), 400
    except Exception as e:
        logger.error(f"Audit endpoint error: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to retrieve audit logs"
        }), 500


@app.route("/translate/<namaste_code>", methods=["GET"])
def map_namaste_code(namaste_code):
    """
    Map a NAMASTE code to ICD-11 TM2 codes.
    
    Args:
        namaste_code (str): NAMASTE diagnostic code from URL path
        
    Returns:
        JSON response with mapping results
    """
    try:
        mappings = main.namaste_to_icd(namaste_code)
        
        if mappings is None:
            return jsonify({
                "error": "not_found",
                "message": f"NAMASTE code '{namaste_code}' not found"
            }), 404
            
        return jsonify({
            "namaste_code": namaste_code,
            "mappings": mappings,
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"Error mapping code {namaste_code}: {e}")
        return jsonify({
            "error": "internal_error",
            "message": "Failed to process mapping request"
        }), 500


@app.route("/condition", methods=["POST"])
def create_fhir_condition():
    """
    Create a FHIR Condition resource from NAMASTE code.
    
    Expected JSON payload:
    {
        "namaste_code": "string",
        "subject_id": "string (optional)",
        "resource_id": "string (optional)"
    }
    
    Returns:
        FHIR Condition resource as JSON
    """
    try:
        data = request.get_json()
        
        if not data or 'namaste_code' not in data:
            return jsonify({
                "error": "bad_request",
                "message": "Missing required field: namaste_code"
            }), 400
            
        namaste_code = data['namaste_code']
        subject_id = data.get('subject_id')
        resource_id = data.get('resource_id')
        
        fhir_condition = main.make_fhir_condition(
            namaste_code=namaste_code,
            subject_id=subject_id,
            resource_id=resource_id
        )
        print(fhir_condition)
        
        return fhir_condition, 200, {'Content-Type': 'application/fhir+json'}
        
    except Exception as e:
        logger.error(f"Error creating FHIR condition: {e}")
        return jsonify({
            "error": "internal_error",
            "message": "Failed to create FHIR condition"
        }), 500

@app.route("/codesystem/<system>", methods=["GET"])
def get_codesystem(system):
    """
    Retrieve code system details.

    Args:
        system (str): Code system identifier from URL path

    Returns:
        JSON response with code system details or error
    """
    if system.lower() == "namaste":
        codesystem = main.codesystem_namaste
    elif system.lower() == "icd11":
        codesystem = main.codesystem_icd
    else:
        codesystem = None
    if codesystem is None:
        return jsonify({
            "error": "not_found",
            "message": f"Code system '{system}' not found"
        }), 404
    try:
        return jsonify(codesystem)
    except Exception as e:
        logger.error(f"Error retrieving code system '{system}': {e}")
        return jsonify({
            "error": "internal_error",
            "message": "Failed to retrieve code system"
        }), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "not_found",
        "message": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        "error": "internal_error",
        "message": "Internal server error"
    }), 500


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting NAMASTE-ICD11 Mapping API server...")
    app.run(debug=False, host='0.0.0.0', port=5000)