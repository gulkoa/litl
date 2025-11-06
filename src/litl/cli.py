import argparse
import os
import sys
import json
from importlib import metadata as importlib_metadata

import rich
from rich.console import Console
from rich.progress import Progress

from .run import compress, decompress, evaluate
from .dot_litl import DotLitl

def parse_args():
    parser = argparse.ArgumentParser(description="Neat CLI for lossy compression via litl")
    parser.add_argument("-m", "--metrics_path", help="Where to save the metrics json")
    parser.add_argument("--version", action="version", version=f"litl v{importlib_metadata.version('litl')}", help="Show the version of litl")

    subparsers = parser.add_subparsers(dest="command", required=True)

    compress_parser = subparsers.add_parser("compress", help="Compress data into a .litl file")
    compress_parser.add_argument("compressor", help="Compressor to use")
    compress_parser.add_argument("original_path", help="Where to find input data to compress")
    compress_parser.add_argument("config_path", help="Where to find config json")
    compress_parser.add_argument("compressed_path", help="Where to save the compressed litl file")

    decompress_parser = subparsers.add_parser("decompress", help="Decompress a .litl file")
    decompress_parser.add_argument("compressed_path", help="Where to find the compressed litl file")
    decompress_parser.add_argument("decompressed_path", help="Where to save the decompressed data")

    decompress_parser.add_argument("-o", "--original_path", help="Where to find original data to compare against, optional")

    evaluate_parser = subparsers.add_parser("evaluate", help="Compress and decompress data, while evaluating the compression")
    evaluate_parser.add_argument("compressor", help="Compressor to use")
    evaluate_parser.add_argument("original_path", help="Where to find input data to compress")
    evaluate_parser.add_argument("config_path", help="Where to find config json")

    evaluate_parser.add_argument("-c", "--compressed_path", help="Where to save the intermediate compressed .litl file")
    evaluate_parser.add_argument("-d", "--decompressed_path", help="Where to save the decompressed data")

    info_parser = subparsers.add_parser("info", help="Print metadata stored in the .litl file")
    info_parser.add_argument("input_path", help="Where to find the .litl file")

    return parser.parse_args()

# ascii_art = \
# """
# ┓• ┓
# ┃┓╋┃
# ┗┗┗┗
# """.strip()
ascii_art = \
"""                                       
  █                    █                  
▄▄█                    █▄▄                
   ██    ██ █████ ██                      
   ██       ██    ██                      
   ██    ██ ██    ██                      
   ██    ██ ██    ██                      
   ██    ██ ██    ██                      
   █████ ██ ██    █████                   
▀▀█                    █▀▀                
  █                    █                                                                   
"""

def main():
    console = Console()
    args = parse_args()
    # fancy rich printing
    console.print(f"[bold green]{ascii_art}[/bold green]")
    console.print(f"[bold]Running litl [green]{args.command}[/green][/bold]")

    try:
        
      if args.command == "compress":
          with open(args.config_path, "r") as f:
              config = f.read()
          config = json.loads(config)
              
          with console.status("[bold green]Compressing...[/bold green]"):
            compressed_blob, meta, metrics = compress(args.compressor, args.original_path, config=config, litl_file_path=args.compressed_path)
            
          console.print(f"Compressed data saved to {args.compressed_path}")
          console.print(f"Metrics: {metrics}")

      elif args.command == "decompress":

          with console.status("[bold green]Decompressing...[/bold green]"):
            decompressed_data, metrics = decompress(litl_file_path=args.compressed_path, decompressed_path=args.decompressed_path, original_data_path=args.original_path)

          console.print(f"Decompressed data saved to {args.decompressed_path}")
          console.print(f"Metrics: {metrics}")

      elif args.command == "evaluate":
          
          with open(args.config_path, "r") as f:
              config = f.read()
          config = json.loads(config)
          
          with console.status("[bold green]Evaluating compression...[/bold green]"):
            decompressed_data, metrics = evaluate(args.compressor, args.original_path, config=config, decompressed_path=args.decompressed_path, litl_file_path=args.compressed_path)
          
          if args.compressed_path:
              console.print(f"Compressed data saved to {args.compressed_path}")

          if args.decompressed_path:
              console.print(f"Decompressed data saved to {args.decompressed_path}")

          console.print("Metrics:", metrics)

      elif args.command == "info":
          
          with console.status("[bold green]Reading metadata...[/bold green]"):
            compressed_blob, compressor_name, compressor_version, meta, litl_version = DotLitl.read(args.input_path)
            
          console.print(f"File {args.input_path} was compressed with {compressor_name} v{compressor_version} using litl v{litl_version}")
          console.print("Metadata:", meta)

      else:
          raise ValueError("Invalid command. Use 'compress', 'decompress', or 'evaluate'.")

      if args.metrics_path is not None:
          with open(args.metrics_path, "w") as f:
              f.write(str(metrics))
          console.print(f"Metrics saved to {args.metrics_path}")
      
    except Exception as e:
        console.print_exception(show_locals=True, width=console.width)
        sys.exit(1)
        
    return 0