from datetime import datetime
from googlemaps.addressvalidation import addressvalidation
import googlemaps
import json
from langchain.agents import tool
import logging
from math import radians, sin, cos, sqrt, atan2
import requests
from typing import Dict, List, Optional, Any
from app.utils import gcp_utils
import stripe

# Setting up logging configuration
logging.basicConfig(level=logging.DEBUG)


##### <LOCALIZA UBICACION> #####
maps_token = gcp_utils.fetch_secret('google-maps-api-key')
gmaps = googlemaps.Client(key=maps_token)

cal_api_key = gcp_utils.fetch_secret('cal-api-key')
API_CAL_BASE_URL = "https://api.cal.com/v2"

# Load places data
def load_places(file_path: str = 'app/utils/semantiks_preferred_spots_addresses.json') -> List[dict]:
    """Load and return the Semantiks preferred spots from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

# Load places data into a global variable
semantiks_places = load_places()

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on Earth."""
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

@tool
def get_closest_location(address: str, places: List[dict] = semantiks_places) -> str:
    """
    Find the nearest Semantiks preferred spot based on a provided address.
    
    Args:
        address: Input address or coordinates
        places: List of Semantiks preferred spots (defaults to loaded places)
        
    Returns:
        Formatted string with closest spot info and directions
    """
    try:
        # Get coordinates for the input address
        geocode = gmaps.geocode(address)[0]['geometry']['location']
        origin = (geocode['lat'], geocode['lng'])
        
        # Find closest place using list comprehension
        distances = [
            (place, haversine_distance(origin[0], origin[1], 
                                     place['location']['latitude'], 
                                     place['location']['longitude']))
            for place in places
            if place.get('location')
        ]
        
        if not distances:
            return "No places found in the database."
            
        # Get closest place
        closest_place, distance = min(distances, key=lambda x: x[1])
        destination = (closest_place['location']['latitude'], 
                      closest_place['location']['longitude'])
        
        # Get directions
        directions = gmaps.directions(
            origin,
            destination,
            mode="driving",
            units="metric"
        )
        
        # Get URL from directions result
        directions_url = f"https://www.google.com/maps/dir/?api=1&origin={origin[0]},{origin[1]}&destination={destination[0]},{destination[1]}"
        
        return (
            f"Closest Semantiks spot:\n"
            f"📍 {closest_place['displayName']}\n"
            f"📮 {closest_place['shortFormattedAddress']}\n"
            f"📏 {distance:.2f} km away\n"
            f"🚗 ETA: {directions[0]['legs'][0]['duration']['text']}\n"
            f"🗺️ Directions: {directions_url}"
        )
        
    except Exception as e:
        return f"Error finding closest location: {str(e)}"
##### </LOCALIZA UBICACION> #####


#### VALIDA UBICACION ####
@tool
def validate_address(address: str) -> str:
    """
    Validate an address using Google Address Validation API

    Args:
        address(str): String of the address.

    Returns:
        str: Formatted address if complete, or "The address is incomplete"
    """
    url = 'https://addressvalidation.googleapis.com/v1:validateAddress'

    # Prepare address payload
    _address = {
        "regionCode": "MX",
        "addressLines": [address]
    }

    # Request headers
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': maps_token,
        'X-Goog-FieldMask': '*'
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json={"address": _address}
        )
        response.raise_for_status()
        result = response.json()

        # Check if address is complete and return formatted address
        if result.get('result', {}).get('verdict', {}).get('addressComplete', False):
            return result.get('result', {}).get('address', {}).get('formattedAddress', '')
        return "The address is incomplete"

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        raise
#### VALIDA UBICACION ###

##### <VALIDA CLIENTE> #####
gigstack_token = gcp_utils.fetch_secret('gigstack-api-key')

@tool
def get_client_id(email: str) -> str:
    """
    Retrieves the client ID associated with a given email address from Gigstack.
    
    Args:
        email (str): The email address to search for in Gigstack.
    
    Returns:
        str: The client ID found for the given email address.
    
    Raises:
        ValueError: If no client is found for the provided email address.
    """
    clients_url = f"https://gigstack-cfdi-bjekv7t4.uc.gateway.dev/v1/clients/list?email={email}"
    headers = {'Authorization': f"Bearer {gigstack_token}"}

    try:
        logging.info(f"Fetching client ID for email: {email}")
        response = requests.get(clients_url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if not data['clients']:
            logging.warning(f"No client found for email: {email}")
            raise ValueError("Error: The email provided does not have a client associated")

        client_id = data['clients'][0]['id']
        logging.info(f"Client ID found: {client_id}")
        return client_id

    except requests.RequestException as e:
        logging.error(f"Error fetching client ID: {str(e)}")
        raise
##### </VALIDA CLIENTE> #####

##### <COMPARTE INFORMACION FACTURAS> #####
def extract_invoices_details(invoices: List[Dict]) -> List[Dict]:
    """
    Extracts essential information from a list of invoice dictionaries.
    Args:
        invoices (list): List of invoice dictionaries.
    Returns:
        list: List of simplified dictionaries containing extracted invoice information.
    """
    extracted_invoices = []
    for invoice in invoices:
        extracted = {}
        processor = invoice.get('processor', '').lower()
        has_stripe = invoice.get('hasStripe', False)
        invoice_processor = 'stripe' if (processor == 'stripe' or has_stripe) else 'gigstack'
        extracted['processor'] = invoice_processor
        extracted['invoice_id'] = invoice.get('id')
        extracted['client_name'] = invoice.get('customer', {}).get('legal_name')
        extracted['amount'] = invoice.get('total')
        extracted['currency'] = invoice.get('currency')
        extracted['status'] = invoice.get('status')
        extracted['created_at'] = invoice.get('date')
        extracted['verification_url'] = invoice.get('verification_url')
        extracted['pdf_url'] = invoice.get('pdf') or invoice.get('pdf_custom_section')
        extracted['payments'] = invoice.get('payments', [])
        items = invoice.get('items') or invoice.get('internalItems') or []
        extracted['items'] = []
        for item in items:
            if invoice_processor == 'stripe' and 'product' in item:
                description = item['product'].get('description') or item['product'].get('unit_name')
            else:
                description = item.get('description') or item.get('name')
            quantity = item.get('quantity', 0)
            total = item.get('total', 0)
            item_info = {
                'description': description.strip() if isinstance(description, str) else description,
                'quantity': quantity,
                'total': total
            }
            extracted['items'].append(item_info)
        extracted_invoices.append(extracted)
    return extracted_invoices

@tool
def get_invoices_details(
    client_id: str,
    limit: int = 3
) -> dict:
    """
    Retrieve invoice details with an optional limit.

    Args:
        client_id (str): The ID of the client to fetch invoices for (required).
        limit (int, optional): The number of invoices to retrieve. Defaults to 3.

    Returns:
        dict: A dictionary containing invoice details and current timestamp.
    """
    base_url = "https://gigstack-cfdi-bjekv7t4.uc.gateway.dev/v1/invoices/list"
    headers = {'Authorization': f"Bearer {gigstack_token}"}
    
    params = {
        'limit': limit,
        'clientId': client_id
    }

    try:
        logging.info(f"Fetching invoice details. Limit: {limit}, Client ID: {client_id}")
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        invoices = extract_invoices_details(data.get('data', []))

        logging.info(f"Successfully retrieved {len(invoices)} invoice(s)")
        return invoices

    except requests.RequestException as e:
        logging.error(f"Error fetching invoice details: {str(e)}")
        raise
##### </COMPARTE INFORMACION FACTURAS> #####

##### <COMPARTE INFORMACION PAGOS> #####
def extract_payments_details(payments: List[Dict]) -> List[Dict]:
    """
    Extracts essential information from a list of payment dictionaries.
    Args:
        payments (list): List of payment dictionaries.
    Returns:
        list: List of simplified dictionaries containing extracted payment information.
    """
    extracted_payments = []
    for payment in payments:
        extracted = {}
        processor = payment.get('processor', '').lower()
        has_stripe = payment.get('hasStripe', False)
        payment_processor = 'stripe' if (processor == 'stripe' or has_stripe) else 'gigstack'
        extracted['processor'] = payment_processor
        extracted['payment_id'] = payment.get('fid')
        extracted['client_id'] = payment.get('clientId')
        extracted['amount'] = payment.get('amount')/100
        extracted['currency'] = payment.get('currency', 'MXN')
        extracted['status'] = payment.get('status', 'unknown')
        extracted['date'] = datetime.fromtimestamp(payment.get('timestamp')/1000).date().strftime('%Y-%m-%d')
        extracted['payment_method'] = payment.get('payment_form')
        extracted_payments.append(extracted)
    return extracted_payments

@tool
def get_payments_details(
    client_id: str,
    limit: int = 5,
    status: Optional[str] = None,
    client_company_id: Optional[str] = None
) -> dict:
    """
    Retrieve payment details for a specific client with optional filters.

    Args:
        client_id (str): Identifier of the client (required).
        limit (int, optional): The number of payments to retrieve. Defaults to 5.
        status (str, optional): Status of the payments to filter (e.g., 'succeeded').
        client_company_id (str, optional): Identifier of the client's company.

    Returns:
        dict: A dictionary containing payment details.
    """
    base_url = "https://gigstack-cfdi-bjekv7t4.uc.gateway.dev/v1/payments/list"
    headers = {'Authorization': f"Bearer {gigstack_token}"}
    
    params = {
        'clientId': client_id,
        'limit': min(limit, 5)
    }
    
    if status:
        params['status'] = status
    if client_company_id:
        params['clientCompanyId'] = client_company_id

    try:
        logging.info(f"Fetching payment details for client ID: {client_id}")
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        payments = extract_payments_details(data.get('data', []))

        logging.info(f"Successfully retrieved {len(payments)} payment(s) for client ID: {client_id}")
        return payments

    except requests.RequestException as e:
        logging.error(f"Error fetching payment details: {str(e)}")
        raise
##### <COMPARTE INFORMACION PAGOS> #####



##### <AGENDA CITA> #####
@tool
def get_available_slots(start_time: str, end_time:str, event_type_id:str, event_type_slug:str):
    """
    Retrieves available time slots for a specific event type within a given time range.

    Args:
        start_time (str): The start of the time range in ISO 8601 format
        end_time (str): The end of the time range in ISO 8601 format
        event_type_id (str): The unique identifier of the event type
        event_type_slug (str): The URL-friendly slug of the event type

    Returns:
        dict: JSON response containing available time slots

    Raises:
        requests.exceptions.HTTPError: If the API request fails

    Example:
        >>> slots = get_available_slots(
        ...     start_time="2024-12-10T00:00:00Z",
        ...     end_time="2024-12-10T23:59:59Z",
        ...     event_type_id="evt_123",
        ...     event_type_slug="30min-meeting"
        ... )
    """
    url = f"{API_CAL_BASE_URL}/slots/available"
    headers = {
        "Authorization": f"Bearer {cal_api_key}",
    }
    querystring = {
        "startTime": start_time,
        "endTime": end_time,
        "eventTypeId": event_type_id,
        "eventTypeSlug": event_type_slug,
        "slotFormat": "range",
    }
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

@tool
def create_booking(start_time:str, event_type_id:int, attendee_name:str, attendee_email:str, attendee_timezone:str):
    """
    Creates a booking through the Calendar API.

    Args:
        start_time (str): The start time of the booking in ISO 8601 format
        event_type_id (int): The unique identifier of the event type
        attendee_name (str): The name of the attendee
        attendee_email (str): The email address of the attendee
        attendee_timezone (str): The timezone of the attendee (e.g., 'America/Mexico_City')

    Returns:
        dict: The JSON response from the API if successful

    Raises:
        requests.exceptions.HTTPError: If the API request fails
    """
    url = f"{API_CAL_BASE_URL}/bookings"
    headers = {
        "Authorization": f"Bearer {cal_api_key}",
        "cal-api-version": "2024-08-13",
        "Content-Type": "application/json"
    }
    payload = {
        "start": start_time,
        "eventTypeId": int(event_type_id),
        "attendee": {
            "name": attendee_name,
            "email": attendee_email,
            "timeZone": attendee_timezone
        }
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 201:
        return response.json()
    else:
        response.raise_for_status()
##### </AGENDA CITA> #####

##### <CALCULA RFC> #####
nufi_token = gcp_utils.fetch_secret('nufi-api-key')

@tool
def calculate_rfc(
    first_name: str, 
    first_surname: str, 
    second_surname: str, 
    birth_date: str, 
    middle_name: str = ""
) -> Dict[str, Any]:
    """
    Calculate RFC with homoclave using Nufi's API.
    Parameters:
    ----------
    first_name : str
        First name of the individual.
    middle_name : str, optional
        Middle name of the individual (default is an empty string).
    first_surname : str
        First surname (apellido paterno).
    second_surname : str
        Second surname (apellido materno).
    birth_date : str
        Birth date of the individual in 'DD/MM/YYYY' format.
    Returns:
    -------
    Dict[str, Any]
        Response from the API containing the calculated RFC, or an error message.
    Raises:
    ------
    ValueError:
        If input validation fails.
    """
    # Sanitize inputs
    first_name = first_name.strip()
    first_surname = first_surname.strip()
    second_surname = second_surname.strip()
    birth_date = birth_date.strip()
    middle_name = middle_name.strip()

    # Validate required inputs
    if not all([first_name, first_surname, second_surname, birth_date]):
        raise ValueError("All inputs (first_name, first_surname, second_surname, birth_date) are required.")
    
    # Validate date format
    try:
        datetime.strptime(birth_date, '%d/%m/%Y')
    except ValueError:
        raise ValueError("birth_date must be in DD/MM/YYYY format")

    # Combine first_name and middle_name into nombres
    nombres = f"{first_name} {middle_name}".strip()
    
    url = "https://nufi.azure-api.net/api/v1/calcular_rfc"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Ocp-Apim-Subscription-Key": nufi_token
    }
    
    payload = {
        "nombres": nombres,
        "apellido_paterno": first_surname,
        "apellido_materno": second_surname,
        "fecha_nacimiento": birth_date
    }
    
    try:
        # Make the API request with timeout
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error: {e.response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
##### </CALCULA RFC> #####

##### <CALCULA CURP> #####

@tool
def calculate_curp(
    first_name: str,
    first_surname: str,
    second_surname: str,
    birth_date: str,
    gender: str,
    birth_state: str,
    middle_name: str = ""
) -> Dict[str, Any]:
    """
    Calculate CURP using Nufi's API.

    Parameters:
    ----------
    first_name : str
        First name of the individual.
    middle_name : str, optional
        Middle name of the individual (default is an empty string).
    first_surname : str
        First surname (apellido paterno).
    second_surname : str
        Second surname (apellido materno).
    birth_date : str
        Birth date of the individual in 'DD/MM/YYYY' format.
    gender : str
        Gender of the individual ('H' for male, 'M' for female).
    birth_state : str
        Two-letter code of the state where the person was born (e.g., 'MN' for Michoacán).

    Returns:
    -------
    Dict[str, Any]
        Response from the API containing the calculated CURP, or an error message.

    Raises:
    ------
    ValueError:
        If input validation fails.
    """
    # Sanitize inputs
    first_name = first_name.strip().upper()
    first_surname = first_surname.strip().upper()
    second_surname = second_surname.strip().upper()
    birth_date = birth_date.strip()
    middle_name = middle_name.strip().upper()
    gender = gender.strip().upper()
    birth_state = birth_state.strip().upper()

    # Validate required inputs
    if not all([first_name, first_surname, second_surname, birth_date, gender, birth_state]):
        raise ValueError("All inputs (except middle_name) are required.")

    # Validate gender
    if gender not in ['H', 'M']:
        raise ValueError("gender must be 'H' for male or 'M' for female")

    # Validate date format and extract components
    try:
        date_obj = datetime.strptime(birth_date, '%d/%m/%Y')
        day = date_obj.strftime('%d')
        month = date_obj.strftime('%m')
        year = date_obj.strftime('%Y')
    except ValueError:
        raise ValueError("birth_date must be in DD/MM/YYYY format")

    # Combine first_name and middle_name
    nombres = f"{first_name} {middle_name}".strip()

    url = "https://nufi.azure-api.net/curp/v1/consulta"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Ocp-Apim-Subscription-Key": nufi_token
    }

    payload = {
        "tipo_busqueda": "datos",
        "clave_entidad": birth_state,
        "dia_nacimiento": day,
        "mes_nacimiento": month,
        "anio_nacimiento": year,
        "nombres": nombres,
        "primer_apellido": first_surname,
        "segundo_apellido": second_surname,
        "sexo": gender
    }

    try:
        # Make the API request with timeout
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error: {e.response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
##### </CALCULA CURP> #####


#### PAYMENT LINKS STRIPE #####
@tool
def get_payment_links_with_products() -> List[Dict[str, Any]]:
    """
    Fetches all payment links and enriches them with product information.
    Returns a list of dictionaries containing payment link and product details.
    """
    stripe.api_key = gcp_utils.fetch_secret('stripe-api-key')
    enriched_products = []
    
    # Get all payment links
    payment_links = stripe.PaymentLink.list()
    
    # Process each payment link
    for link in payment_links.data:
        # Get line items for this payment link
        line_items = stripe.PaymentLink.list_line_items(link.id)
        
        # Process each line item
        for item in line_items.data:
            # Get the product ID from the price object
            product_id = item.price.product
            
            # Get the full product information
            product = stripe.Product.retrieve(product_id)
            
            # Create an enriched product object with the payment link
            product_info = {
                "id": product.id,
                "name": product.name,
                "description": product.description,
                "active": product.active,
                "created": product.created,
                "type": product.type,
                "default_price": product.default_price,
                "PAYMENT_LINK": link.url,
                "price_amount": item.price.unit_amount / 100,  # Convert from cents
                "currency": item.currency,
                "billing_frequency": item.price.recurring.interval if hasattr(item.price, 'recurring') else None,
                "trial_period_days": link.subscription_data.get('trial_period_days')
            }
            
            enriched_products.append(product_info)
    
    return enriched_products
#### </PAYMENT LINKS STRIPE> #####
