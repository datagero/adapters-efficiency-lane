def extract_after_slash(input_string):
    parts = input_string.split('/')  # Split at the first '/' only
    if len(parts) > 1:
        return parts[-1]  # Return the part after the last '/'
    else:
        return input_string  # Return the original string if no '/' is found