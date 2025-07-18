import os
import re

def add_unique_keys_to_file(file_path, component_name):
    """Add unique keys to Streamlit elements in a file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to find elements without keys
    patterns = [
        (r'st\.selectbox\((.*?)\)', f'st.selectbox(\\1, key="{component_name}_selectbox_{{}}"))'),
        (r'st\.slider\((.*?)\)', f'st.slider(\\1, key="{component_name}_slider_{{}}"))'),
        (r'st\.checkbox\((.*?)\)', f'st.checkbox(\\1, key="{component_name}_checkbox_{{}}"))'),
        (r'st\.text_input\((.*?)\)', f'st.text_input(\\1, key="{component_name}_text_input_{{}}"))'),
        (r'st\.text_area\((.*?)\)', f'st.text_area(\\1, key="{component_name}_text_area_{{}}"))'),
        (r'st\.multiselect\((.*?)\)', f'st.multiselect(\\1, key="{component_name}_multiselect_{{}}"))'),
    ]
    
    for pattern, replacement in patterns:
        matches = re.finditer(pattern, content)
        for i, match in enumerate(matches):
            if 'key=' not in match.group(0):
                content = content.replace(match.group(0), replacement.format(i))
    
    with open(file_path, 'w') as f:
        f.write(content)

# Run for all components
components = [
    ('frontend/components/sidebar.py', 'sidebar'),
    ('frontend/components/enhanced_voice_chat.py', 'enhanced_voice'),
    ('frontend/components/voice_chat.py', 'voice'),
    ('frontend/components/advanced_search.py', 'search'),
    ('frontend/components/query_refinement.py', 'refinement'),
    ('frontend/components/document_upload.py', 'upload'),
    ('frontend/components/enhanced_chat.py', 'chat'),
]

for file_path, component_name in components:
    if os.path.exists(file_path):
        add_unique_keys_to_file(file_path, component_name)
        print(f"Fixed keys in {file_path}")
