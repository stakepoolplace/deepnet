package dmonner.xlbp;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;
import java.util.Collection;

import dmonner.xlbp.compound.Compound;
import dmonner.xlbp.compound.WeightedCompound;
import dmonner.xlbp.util.ListMap;
import dmonner.xlbp.util.ReflectionTools;

public class NetworkConfigurator
{
	public enum ConnectionType
	{
		DIRECT, WEIGHTED
	}

	private final String specSep = "\\s+";
	private final String[] pkgs = { "dmonner.xlbp", "dmonner.xlbp.compound", "dmonner.xlbp.layer" };
	private final String paramPrefix = "Param-";
	private final String componentPrefix = "Component-";
	private final String[] connectPrefixes = { "Connect-Direct-", "Connect-Weighted-",
			"Connect-WeightedAll-" };
	private final String networkPrefix = "Network-";
	private final String activatePrefix = "-";
	private final String trainPrefix = "+";
	private final String entryPrefix = ">";
	private final String exitPrefix = "<";

	private final ListMap<String, Object> params;
	private final ListMap<String, Component> components;
	private final ListMap<String, Component> specified;
	private final ListMap<String, Network> networks;
	private final Network meta;
	private final WeightUpdaterType wut;
	private final WeightInitializer win;

	public NetworkConfigurator(final ListMap<String, Object> params)
	{
		this.params = params;
		this.components = new ListMap<String, Component>();
		this.specified = new ListMap<String, Component>();
		this.networks = new ListMap<String, Network>();

		// Create all the necessary Components
		for(final String key : params.keyList())
			if(key.startsWith(componentPrefix))
				addComponent(key.substring(componentPrefix.length()), params.get(key).toString());

		// Connect the Components up as specified
		for(final String key : params.keyList())
			for(final String prefix : connectPrefixes)
				if(key.startsWith(prefix) && asBoolean(params.get(key)))
					addConnection(prefix, key.substring(prefix.length()));

		// Create an "all" network to build everything (even if the order is wrong)
		final float connectionProbability = (Float) params.get("connectionProbability");
		final String updaterType = params.get("updaterType").toString();
		win = new UniformWeightInitializer(connectionProbability);
		wut = WeightUpdaterType.fromString(updaterType, params);

		meta = new Network("Meta");
		meta.setWeightInitializer(win);
		meta.setWeightUpdaterType(wut);
		for(final Component comp : specified.values())
			meta.add(comp);

		// Put the Components in their Networks
		for(final String key : params.keyList())
			if(key.startsWith(networkPrefix))
				addNetwork(key.substring(networkPrefix.length()), params.get(key).toString());
	}

	public void addComponent(final String name, final String spec)
	{
		if(spec.trim().isEmpty())
			return;

		final String[] part = spec.split(specSep);
		if(part.length < 1)
			throw new IllegalArgumentException("Uninterpretable Component definition: " + spec);
		final String classname = part[0];
		final Class<?> clazz = ReflectionTools.findClass(classname, pkgs);

		// build arrays of signature types and actual arguments
		final Class<?>[] sign = new Class<?>[part.length];
		final Object[] args = new Object[part.length];
		sign[0] = String.class;
		args[0] = name;
		for(int i = 1; i < args.length; i++)
		{
			args[i] = findParam(part[i]);
			sign[i] = ReflectionTools.unbox(args[i].getClass());
		}

		try
		{
			final Constructor<?> constr = ReflectionTools.findConstructor(clazz, sign);
			final Object inst = constr.newInstance(args);
			specified.put(name, (Component) inst);
			components.put(name, (Component) inst);
			if(inst instanceof Compound)
				for(final Component sub : ((Compound) inst).getComponents())
					components.put(sub.getName(), sub);
		}
		catch(final InvocationTargetException ex)
		{
			throw new IllegalArgumentException("Exception while running constructor for " + classname
					+ " with signature " + Arrays.deepToString(sign), ex);
		}
		catch(final IllegalAccessException ex)
		{
			ex.printStackTrace();
			throw new IllegalArgumentException("Inaccessible constructor for " + classname
					+ " with signature " + Arrays.deepToString(sign), ex);
		}
		catch(final InstantiationException ex)
		{
			ex.printStackTrace();
			throw new IllegalArgumentException("Uninstantiable constructor for " + classname
					+ " with signature " + Arrays.deepToString(sign), ex);
		}
	}

	public void addConnection(final ConnectionType type, final Component from, final Component to)
	{
		if(type == ConnectionType.DIRECT)
			if(to instanceof DownstreamComponent && from instanceof UpstreamComponent)
				((DownstreamComponent) to).addUpstream((UpstreamComponent) from);
			else
				throw new IllegalArgumentException("Cannot create direct connection between " + from
						+ " and " + to);
		else if(type == ConnectionType.WEIGHTED)
			if(to instanceof WeightedCompound && from instanceof UpstreamComponent)
				((WeightedCompound) to).addUpstreamWeights((UpstreamComponent) from);
			else
				throw new IllegalArgumentException("Cannot create weighted connection between " + from
						+ " and " + to);
		else
			throw new IllegalArgumentException("Unhandled ConnectionType: " + type);
	}

	public void addConnection(final String prefix, final String layers)
	{
		ConnectionType type = null;
		int expectedParts;

		if(prefix.equals(connectPrefixes[0]))
		{
			type = ConnectionType.DIRECT;
			expectedParts = 2;
		}
		else if(prefix.equals(connectPrefixes[1]))
		{
			type = ConnectionType.WEIGHTED;
			expectedParts = 2;
		}
		else
		{
			throw new IllegalArgumentException("Unhandled ConnectionType: " + prefix);
		}

		final String[] parts = layers.split("-");

		if(parts.length != expectedParts)
			throw new IllegalArgumentException("Wrong number of layers in connect parameter: " + layers);

		if(!components.containsKey(parts[0]))
			throw new IllegalArgumentException("No from-component named: " + parts[0]);

		if(expectedParts > 1 && !components.containsKey(parts[1]))
			throw new IllegalArgumentException("No to-component named: " + parts[1]);

		Component from = null, to = null;
		from = components.get(parts[0]);
		if(expectedParts > 1)
			to = components.get(parts[1]);

		addConnection(type, from, to);
	}

	public void addNetwork(final String name, final String spec)
	{
		final Network net = new Network(name);
		net.setWeightInitializer(win);
		net.setWeightUpdaterType(wut);

		networks.put(name, net);
		components.put(name, net);

		if(spec.trim().isEmpty())
			return;

		final String[] parts = spec.split(specSep);

		for(String part : parts)
		{
			boolean activate = true;
			boolean train = true;
			boolean entry = false;
			boolean exit = false;

			// Remove prefixes from the part name until there aren't any more.
			String lastpart;
			do
			{
				lastpart = part;
				if(part.startsWith(activatePrefix))
				{
					train = false;
					part = part.substring(activatePrefix.length());
				}
				else if(part.startsWith(trainPrefix))
				{
					activate = false;
					part = part.substring(trainPrefix.length());
				}
				else if(part.startsWith(entryPrefix))
				{
					entry = true;
					part = part.substring(entryPrefix.length());
				}
				else if(part.startsWith(exitPrefix))
				{
					exit = true;
					part = part.substring(exitPrefix.length());
				}
			}
			while(!part.equals(lastpart));

			if(!components.containsKey(part))
				throw new IllegalArgumentException("Cannot find component to add to network " + net + ": "
						+ part);

			net.add(components.get(part), activate, train, entry, exit);
		}
	}

	private boolean asBoolean(final Object obj)
	{
		if(obj instanceof Boolean)
			return (Boolean) obj;
		else if(obj instanceof Integer)
			return ((Integer) obj) != 0;
		return false;
	}

	public void build()
	{
		meta.build();
	}

	private Object findParam(final String key)
	{
		Object val = params.get(key);

		if(val == null)
			val = params.get(paramPrefix + key);

		if(val == null)
			val = components.get(key);

		if(val == null)
			throw new IllegalArgumentException("Variable not found: " + key);

		return val;
	}

	public Component getComponent(final String name)
	{
		return components.get(name);
	}

	public Collection<Component> getComponents()
	{
		return components.values();
	}

	public Network getMetaNetwork()
	{
		return meta;
	}

	public Network getNetwork(final String name)
	{
		return networks.get(name);
	}

	public Collection<Network> getNetworks()
	{
		return networks.values();
	}

	public boolean optimize()
	{
		return meta.optimize();
	}

	@Override
	public String toString()
	{
		return components.toString();
	}
}
