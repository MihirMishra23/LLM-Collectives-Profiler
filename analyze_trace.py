def calculate_collective_bandwidth(comm_events, world_size=4):
    """Calculate bandwidth using alpha-beta model for collective operations."""
    collective_stats = defaultdict(list)
    
    for event in comm_events:
        name = event['name']
        duration_sec = event['duration'] / 1e6
        message_bytes = event.get('bytes', 0)
        
        if duration_sec == 0 or message_bytes == 0:
            continue
        
        # Determine collective type and theoretical data transferred
        if 'AllReduce' in name:
            coll_type = 'AllReduce'
            # Ring AllReduce: 2(S/n)(n-1)
            data_transferred = 2 * message_bytes * (world_size - 1) / world_size
        elif 'ReduceScatter' in name:
            coll_type = 'ReduceScatter'
            # ReduceScatter: (S/n)(n-1)
            data_transferred = message_bytes * (world_size - 1) / world_size
        elif 'AllGather' in name:
            coll_type = 'AllGather'
            # AllGather: (S/n)(n-1)
            data_transferred = message_bytes * (world_size - 1) / world_size
        elif 'Broadcast' in name:
            coll_type = 'Broadcast'
            # Broadcast: S * log2(n)
            data_transferred = message_bytes * math.log2(world_size) if world_size > 1 else message_bytes
        else:
            coll_type = 'Other'
            data_transferred = message_bytes
        
        # Bandwidth in GB/s
        bandwidth_gbps = (data_transferred / 1e9) / duration_sec
        collective_stats[coll_type].append({
            'bandwidth_gbps': bandwidth_gbps,
            'message_bytes': message_bytes,
            'data_transferred': data_transferred,
            'duration_ms': duration_sec * 1000
        })
