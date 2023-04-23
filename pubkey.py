import argparse

import eth_keys
import hexbytes
from eth_account._utils import signing


def get_public_key(tx_raw):
    txn_bytes = hexbytes.HexBytes(tx_raw)
    typed_txn = signing.TypedTransaction.from_bytes(txn_bytes)

    msg_hash = typed_txn.hash()
    hash_bytes = hexbytes.HexBytes(msg_hash)

    vrs = typed_txn.vrs()
    v, r, s = vrs
    v_standard = signing.to_standard_v(v)
    vrs = (v_standard, r, s)

    signature_obj = eth_keys.KeyAPI().Signature(vrs=vrs)
    pubkey = signature_obj.recover_public_key_from_msg_hash(hash_bytes)
    
    return pubkey

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--transaction", type=str, help="Raw Tx Hex", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    tx_raw = args.transaction    
    pubkey = get_public_key(tx_raw)
    print(f"Pubkey: {pubkey}")
