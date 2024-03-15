package dmonner.xlbp.util;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * This class represents a Map that remembers the order of items added to it, like a List would. The
 * keySet and entrySet are ListSets which iterate in the order that items were added to the ListMap.
 * The values is a similarly ordered List. Insertion and containment are still O(log(n)) as in Map,
 * though removal is now O(n) as in List. This class is most useful in instances where arbitrary
 * ordering, insertion, and containment are the primary operations.
 * 
 * @author dmonner
 * 
 * @param <K, V> The key- and value-types to be stored in the ListMap
 */
public class ListMap<K, V> implements Map<K, V>, Serializable
{
	public static class Entry<K, V> implements Map.Entry<K, V>
	{
		private final K key;
		private V value;

		public Entry(final K key, final V value)
		{
			this.key = key;
			this.value = value;
		}

		@Override
		public K getKey()
		{
			return key;
		}

		@Override
		public V getValue()
		{
			return value;
		}

		@Override
		public int hashCode()
		{
			return key.hashCode();
		}

		@Override
		public V setValue(final V value)
		{
			final V rv = this.value;
			this.value = value;
			return rv;
		}
	}

	private static final long serialVersionUID = 1L;

	private final List<K> keys;
	private final Map<K, V> map;

	public ListMap()
	{
		keys = new LinkedList<K>();
		map = new HashMap<K, V>();
	}

	public ListMap(final ListMap<K, V> other)
	{
		keys = new LinkedList<K>(other.keys);
		map = new HashMap<K, V>(other.map);
	}

	public ListMap(final Map<K, V> other)
	{
		keys = new LinkedList<K>(other.keySet());
		map = new HashMap<K, V>(other);
	}

	@Override
	public void clear()
	{
		keys.clear();
		map.clear();
	}

	@Override
	public boolean containsKey(final Object key)
	{
		return map.containsKey(key);
	}

	@Override
	public boolean containsValue(final Object value)
	{
		return map.containsValue(value);
	}

	@Override
	public Set<Map.Entry<K, V>> entrySet()
	{
		final Set<Map.Entry<K, V>> set = new ListSet<Map.Entry<K, V>>();

		for(final K key : keys)
			set.add(new Entry<K, V>(key, map.get(key)));

		return set;
	}

	@Override
	public V get(final Object key)
	{
		return map.get(key);
	}

	public K getKey(final int index)
	{
		return keys.get(index);
	}

	@Override
	public boolean isEmpty()
	{
		return map.isEmpty();
	}

	public List<K> keyList()
	{
		final List<K> list = new ArrayList<K>();

		for(final K key : keys)
			list.add(key);

		return list;
	}

	@Override
	public ListSet<K> keySet()
	{
		final ListSet<K> set = new ListSet<K>();

		for(final K key : keys)
			set.add(key);

		return set;
	}

	@Override
	public V put(final K key, final V value)
	{
		if(!map.containsKey(key))
			keys.add(key);

		return map.put(key, value);
	}

	@Override
	public void putAll(final Map<? extends K, ? extends V> m)
	{
		for(final K key : m.keySet())
			put(key, m.get(key));
	}

	@Override
	public V remove(final Object key)
	{
		keys.remove(key);
		return map.remove(key);
	}

	@Override
	public int size()
	{
		return map.size();
	}

	@Override
	public List<V> values()
	{
		final List<V> set = new LinkedList<V>();

		for(final K key : keys)
			set.add(map.get(key));

		return set;
	}

}
