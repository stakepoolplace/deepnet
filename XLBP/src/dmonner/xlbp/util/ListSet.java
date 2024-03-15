package dmonner.xlbp.util;

import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;
import java.util.TreeSet;

/**
 * This class represents a Set that remembers the order of items added to it, like a List would.
 * This set's Iterator is a ListIterator and returns items in the order in which they were added.
 * Yet, the set preserves uniqueness, unlike a list. Insertion and containment are still O(log(n))
 * as in Set, though removal is now O(n) as in List. This class is most useful in instances where
 * arbitrary ordering, insertion, and containment are the primary operations.
 * 
 * @author dmonner
 * 
 * @param <E>
 *          The type to be stored in the ListSet
 */
public class ListSet<E> implements Set<E>, List<E>
{
	private final Set<E> set;
	private final List<E> list;

	public ListSet()
	{
		this.set = new TreeSet<E>();
		this.list = new LinkedList<E>();
	}

	private ListSet(final List<E> list, final Set<E> set)
	{
		this.set = set;
		this.list = list;
	}

	public ListSet(final ListSet<E> other)
	{
		this.set = new TreeSet<E>(other.set);
		this.list = new LinkedList<E>(other.list);
	}

	public ListSet(final Set<E> other)
	{
		this.set = new TreeSet<E>(other);
		this.list = new LinkedList<E>(other);
	}

	@Override
	public boolean add(final E e)
	{
		if(!set.contains(e))
		{
			list.add(e);
			return set.add(e);
		}

		return false;
	}

	@Override
	public void add(final int index, final E element)
	{
		if(!set.contains(element))
		{
			list.add(index, element);
			set.add(element);
		}
	}

	@Override
	public boolean addAll(final Collection<? extends E> c)
	{
		boolean changed = false;

		for(final E e : c)
			if(add(e))
				changed = true;

		return changed;
	}

	@Override
	public boolean addAll(final int index, final Collection<? extends E> c)
	{
		int inserted = 0;

		for(final E e : c)
		{
			if(!set.contains(e))
			{
				list.add(index + inserted, e);
				set.add(e);
				inserted++;
			}
		}

		return inserted > 0;
	}

	@Override
	public void clear()
	{
		set.clear();
		list.clear();
	}

	@Override
	public boolean contains(final Object o)
	{
		return set.contains(o);
	}

	@Override
	public boolean containsAll(final Collection<?> c)
	{
		return set.containsAll(c);
	}

	@Override
	public E get(final int index)
	{
		return list.get(index);
	}

	@Override
	public int indexOf(final Object o)
	{
		return list.indexOf(o);
	}

	@Override
	public boolean isEmpty()
	{
		return set.isEmpty();
	}

	@Override
	public Iterator<E> iterator()
	{
		return list.iterator();
	}

	@Override
	public int lastIndexOf(final Object o)
	{
		return list.lastIndexOf(o);
	}

	@Override
	public ListIterator<E> listIterator()
	{
		return list.listIterator();
	}

	@Override
	public ListIterator<E> listIterator(final int index)
	{
		return list.listIterator(index);
	}

	@Override
	public E remove(final int index)
	{
		final E e = list.remove(index);

		if(e != null)
			set.remove(e);

		return e;
	}

	@Override
	public boolean remove(final Object o)
	{
		list.remove(o);
		return set.remove(o);
	}

	@Override
	public boolean removeAll(final Collection<?> c)
	{
		list.removeAll(c);
		return set.removeAll(c);
	}

	@Override
	public boolean retainAll(final Collection<?> c)
	{
		list.retainAll(c);
		return set.retainAll(c);
	}

	@Override
	public E set(final int index, final E element)
	{
		final E old = list.set(index, element);
		set.remove(old);
		set.add(element);
		return old;
	}

	@Override
	public int size()
	{
		return set.size();
	}

	@Override
	public ListSet<E> subList(final int fromIndex, final int toIndex)
	{
		final List<E> sublist = list.subList(fromIndex, toIndex);
		final Set<E> subset = new TreeSet<E>(set);
		subset.retainAll(sublist);
		return new ListSet<E>(sublist, subset);
	}

	@Override
	public Object[] toArray()
	{
		return list.toArray();
	}

	@Override
	public <T> T[] toArray(final T[] a)
	{
		return list.toArray(a);
	}

}
