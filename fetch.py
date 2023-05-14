
'''Scrapes midi files referenced by a given webpage.'''

import argparse
import os
import sys
import requests
from bs4 import BeautifulSoup
import urllib.request

def fetch_html(url):
  '''Fetches the html content of a given url.'''
  response = requests.get(url)
  return response.text

def find_mid_links(html):
  '''Finds all links to midi files in the given html.'''
  soup = BeautifulSoup(html, "html.parser")
  links = []
  for link in soup.find_all("a"):
    href = link.get("href")
    if href and href.endswith(".mid"):
      links.append(href)
  return links

def replace_spaces_with_percent20(s):
    return s.replace(" ", "%20")

def download_mid_files(url, links, output_path):
  '''Downloads all midi files referenced by the given links.'''
  for link in links:
    file_name = os.path.join(output_path, os.path.basename(link))
    download_url = urllib.parse.urljoin(url, link).replace(' ', '%20')
    print(f'Downloading {file_name}...')
    try:
      urllib.request.urlretrieve(download_url, file_name)
    except:
      print(f'Failed to download {file_name}.')
      continue

def fetch_midis(url, output_path):
  '''Fetches all midi files referenced by the given url, and places them in the output folder.'''
  html = fetch_html(url)
  mid_links = find_mid_links(html)
  if not mid_links:
    print('No midi files found at the given URL.')
    return
  download_mid_files(url, mid_links, output_path)
  print('Download completed.')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Fetches midi files linked by URL, and places them in the output folder.')
  parser.add_argument('url', help='URL to scan for midi file downloads.')
  parser.add_argument('output', help='path to store downloaded midi files.')
  args = parser.parse_args()
  fetch_midis(args.url, args.output)
